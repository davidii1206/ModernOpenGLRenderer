// Example 34 — MDC on the GPU: the occlusion-free patch-atlas pipeline of
// example 31, ported to compute shaders.
//
// The CPU-only pipeline (gfx::CoverageAtlas) welds, crease-smooths, clusters
// (BFS over shared edges), merges, packs and rasterises. Here every stage that
// is embarrassingly parallel runs on the GPU as GLSL 460 compute shaders:
//
//   crease normals  incident lists via per-vertex atomics + per-corner averages
//   clustering      reformulated as connected components: triangles project
//                   onto per-(submesh, axis) 128x128 depth cells; a cell whose
//                   depth range stays within `epsilon` is occlusion-free, so
//                   every triangle in it unions into one patch (atomicCAS
//                   cell owner + a lock-free union-find fixpoint). No shared
//                   edge map is needed — edge adjacency is implicit in shared
//                   cells.
//   small-patch merge  patches below min_patch_size absorb into the largest
//                   neighbouring patch through cells that stay occlusion-free.
//   rasterisation   one thread per triangle, conservative texel footprint,
//                   atomicMin/Max on a monotonic float->uint transform,
//                   last-writer-wins UV / octahedral normal.
//   mip chains      built level-by-level by re-aggregating each node's region
//                   straight from the atlas arrays (no pyramid stored).
//   BVH             Morton codes + stable LSD radix sort, level-parallel top-
//                   down median split, bottom-up AABB merge, pre-order renumber.
//
// Host (CPU) still owns the cheap, inherently sequential bits: position weld,
// texture sizing / budget fit, and skyline atlas packing.
//
// The result is diffed against the CPU reference pipeline built from the same
// config, then rendered with multi-draw indirect.
//
// Usage: 34_mdc_gpu [model.glb] [texel_density] [budget_Mtexels]
//                  [min_patch_size] [epsilon]

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <chrono>
#include <unordered_map>

// ======================================================================
// constants + host helpers
// ======================================================================

static constexpr uint32_t GRID      = 128;     // cells per (submesh, axis)
static constexpr uint32_t UCELL     = GRID * GRID;
static constexpr uint32_t MAX_FOOT  = 256;     // cell footprint cap per triangle
static constexpr uint32_t MAX_LEVELS = 12;
static constexpr uint32_t MIP_CAP   = 4000000; // max mip nodes across all levels
static constexpr uint32_t FIX_ITERS = 24;      // union-find pointer-jumping passes

static uint32_t f2s(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
static float s2f(uint32_t u) {
    uint32_t v = (u & 0x80000000u) ? (u ^ 0x80000000u) : ~u;
    float f; std::memcpy(&f, &v, 4);
    return f;
}
static int next_pow2_int(int v) {
    int n = 1;
    while (n < v) n <<= 1;
    return n;
}

struct AxisBasis { glm::vec3 d, u, v; };
static AxisBasis axis_basis(int axis) {
    glm::vec3 d(0.0f);
    if (axis < 2)      d = glm::vec3(axis == 0 ? 1.0f : -1.0f, 0.0f, 0.0f);
    else if (axis < 4) d = glm::vec3(0.0f, axis == 2 ? 1.0f : -1.0f, 0.0f);
    else               d = glm::vec3(0.0f, 0.0f, axis == 4 ? 1.0f : -1.0f);
    glm::vec3 u = std::fabs(d.x) > 0.5f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
    glm::vec3 v = glm::normalize(glm::cross(d, u));
    return {d, u, v};
}
static float proj_min(const glm::vec3& mn, const glm::vec3& mx, const glm::vec3& a) {
    return (a.x > 0 ? mn.x : mx.x) * a.x + (a.y > 0 ? mn.y : mx.y) * a.y
         + (a.z > 0 ? mn.z : mx.z) * a.z;
}
static float proj_max(const glm::vec3& mn, const glm::vec3& mx, const glm::vec3& a) {
    return (a.x > 0 ? mx.x : mn.x) * a.x + (a.y > 0 ? mx.y : mn.y) * a.y
         + (a.z > 0 ? mx.z : mn.z) * a.z;
}

// ---- buffer plumbing ----------------------------------------------------

static gl::Buffer make_buf(std::size_t bytes) {
    gl::Buffer b(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    if (bytes) b.data(nullptr, bytes);
    return b;
}
static void fill_u32(gl::Buffer& b, uint32_t val) {
    b.clear(GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &val);
}
static void upload(gl::Buffer& b, const void* p, std::size_t bytes) {
    if (bytes == 0) return;
    if (b.size() == bytes) b.sub_data(p, 0, bytes);
    else b.data(p, bytes);
}
template <typename T>
static void upload(gl::Buffer& b, const std::vector<T>& v) {
    upload(b, v.data(), v.size() * sizeof(T));
}
static void readback(gl::Buffer& b, void* dst, std::size_t bytes) {
    void* p = b.map_range(0, bytes, GL_MAP_READ_BIT);
    if (!p) { std::memset(dst, 0, bytes); return; }
    std::memcpy(dst, p, bytes);
    b.unmap();
}
template <typename T>
static void readback(gl::Buffer& b, std::vector<T>& v) {
    v.resize(b.size() / sizeof(T));
    readback(b, v.data(), b.size());
}

struct Kern {
    gl::Program prog;
};
static Kern make_kern(const std::string& src) {
    Kern k;
    gl::Shader cs(gl::ShaderType::compute, src);
    if (!cs.compiled()) {
        fprintf(stderr, "Compute shader failed:\n%s\nsource:\n%s\n",
                cs.info_log().c_str(), src.c_str());
        std::exit(1);
    }
    k.prog.attach(cs);
    if (!k.prog.link()) {
        fprintf(stderr, "Compute program link failed:\n%s\n", k.prog.info_log().c_str());
        std::exit(1);
    }
    return k;
}
static void dispatch(const Kern& k, uint32_t n) {
    k.prog.use();
    gl::dispatch_compute((n + 255) / 256, 1, 1);
}

// ---- GPU timing ----------------------------------------------------------

static double run_ms(const std::function<void()>& fn) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    gl::Sync fence;
    fence.client_wait(5000000000ULL);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ======================================================================
// common GLSL preamble
// ======================================================================

static const char* PRE = R"(
#version 460 core
#define LOCAL 256
#define GRID 128
#define UCELL (GRID * GRID)
layout(local_size_x = LOCAL) in;
uint f2s(float x){ uint u = floatBitsToUint(x); return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u); }
float s2f(uint u){ uint v = (u & 0x80000000u) != 0u ? (u ^ 0x80000000u) : ~u; return uintBitsToFloat(v); }
const uint SENT = 0xFFFFFFFFu;
void axis_basis(uint axis, out vec3 d, out vec3 u, out vec3 v){
    d = vec3(0.0);
    if (axis < 2u) d.x = (axis == 0u) ? 1.0 : -1.0;
    else if (axis < 4u) d.y = (axis == 2u) ? 1.0 : -1.0;
    else d.z = (axis == 4u) ? 1.0 : -1.0;
    u = (abs(d.x) > 0.5) ? vec3(0,1,0) : vec3(1,0,0);
    v = normalize(cross(d, u));
}
vec2 encOct(vec3 n){
    float l = abs(n.x) + abs(n.y) + abs(n.z);
    if (l < 1e-8) n = vec3(0.0, 0.0, 1.0);
    else n /= l;
    if (n.z < 0.0){
        n.x = (1.0 - abs(n.y)) * (n.x >= 0.0 ? 1.0 : -1.0);
        n.y = (1.0 - abs(n.x)) * (n.y >= 0.0 ? 1.0 : -1.0);
    }
    return vec2(n.x * 0.5 + 0.5, n.y * 0.5 + 0.5);
}
vec3 decOct(vec2 e){
    vec3 n = vec3(e.x * 2.0 - 1.0, e.y * 2.0 - 1.0, 0.0);
    n.z = 1.0 - abs(n.x) - abs(n.y);
    float t = max(-n.z, 0.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}
float proj_min3(float mn0, float mn1, float mn2, float mx0, float mx1, float mx2, vec3 a){
    return (a.x > 0.0 ? mn0 : mx0) * a.x + (a.y > 0.0 ? mn1 : mx1) * a.y
         + (a.z > 0.0 ? mn2 : mx2) * a.z;
}
float proj_max3(float mn0, float mn1, float mn2, float mx0, float mx1, float mx2, vec3 a){
    return (a.x > 0.0 ? mx0 : mn0) * a.x + (a.y > 0.0 ? mx1 : mn1) * a.y
         + (a.z > 0.0 ? mx2 : mn2) * a.z;
}
)";

static std::string kern(const char* body) { return std::string(PRE) + body; }

// ======================================================================
// kernel sources
// ======================================================================

static const char* K_FACE = R"(
layout(binding=0) readonly buffer Pos   { float pos[]; };
layout(binding=1) readonly buffer Idx   { uint  idx[]; };
layout(binding=2) buffer       FaceN   { float faceN[]; };
uniform uint uTri;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uTri) return;
    uint a = idx[3u*g], b = idx[3u*g+1u], c = idx[3u*g+2u];
    vec3 p0 = vec3(pos[3u*a], pos[3u*a+1u], pos[3u*a+2u]);
    vec3 p1 = vec3(pos[3u*b], pos[3u*b+1u], pos[3u*b+2u]);
    vec3 p2 = vec3(pos[3u*c], pos[3u*c+1u], pos[3u*c+2u]);
    vec3 n = cross(p1 - p0, p2 - p0);
    float l = length(n);
    if (l > 1e-12) n /= l; else n = vec3(0, 0, 1);
    faceN[3u*g] = n.x; faceN[3u*g+1u] = n.y; faceN[3u*g+2u] = n.z;
}
)";

static const char* K_TRIPREP = R"(
layout(binding=0) readonly buffer Pos    { float pos[]; };
layout(binding=1) readonly buffer Idx    { uint  idx[]; };
layout(binding=2) readonly buffer TriSub { uint  sub[]; };
layout(binding=3) readonly buffer FaceN  { float faceN[]; };
layout(binding=4) buffer       TriAxis  { uint  axis[]; };
layout(binding=5) buffer       SubAABB  { uint  saabb[]; };
uniform uint uTri;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uTri) return;
    vec3 n = vec3(faceN[3u*g], faceN[3u*g+1u], faceN[3u*g+2u]);
    vec3 an = abs(n);
    uint a;
    if (an.x >= an.y && an.x >= an.z) a = (n.x >= 0.0) ? 0u : 1u;
    else if (an.y >= an.z)            a = (n.y >= 0.0) ? 2u : 3u;
    else                              a = (n.z >= 0.0) ? 4u : 5u;
    axis[g] = a;
    uint base = sub[g] * 6u;
    for (int e = 0; e < 3; ++e) {
        uint vi = idx[3u*g + uint(e)];
        for (int k = 0; k < 3; ++k) {
            float v = pos[3u*vi + uint(k)];
            atomicMin(saabb[base + uint(k)],     f2s(v));
            atomicMax(saabb[base + 3u + uint(k)], f2s(v));
        }
    }
}
)";

static const char* K_INCCOUNT = R"(
layout(binding=0) readonly buffer Idx   { uint idx[]; };
layout(binding=1) buffer       IncPos { uint incPos[]; };
uniform uint uTri;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uTri) return;
    for (int e = 0; e < 3; ++e)
        atomicAdd(incPos[idx[3u*g + uint(e)]], 1u);
}
)";

static const char* K_INCFILL = R"(
layout(binding=0) readonly buffer Idx     { uint  idx[]; };
layout(binding=1) readonly buffer FaceN   { float faceN[]; };
layout(binding=2) readonly buffer IncOff  { uint  incOff[]; };
layout(binding=3) buffer       IncFill { uint  incFill[]; };
layout(binding=4) buffer       IncFace { float incFace[]; };
uniform uint uTri;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uTri) return;
    for (int e = 0; e < 3; ++e) {
        uint v = idx[3u*g + uint(e)];
        uint slot = incOff[v] + atomicAdd(incFill[v], 1u);
        incFace[3u*slot] = faceN[3u*g];
        incFace[3u*slot+1u] = faceN[3u*g+1u];
        incFace[3u*slot+2u] = faceN[3u*g+2u];
    }
}
)";

static const char* K_CREASE = R"(
layout(binding=0) readonly buffer Idx    { uint  idx[]; };
layout(binding=1) readonly buffer FaceN  { float faceN[]; };
layout(binding=2) readonly buffer IncOff { uint  incOff[]; };
layout(binding=3) readonly buffer IncPos { uint  incPos[]; };
layout(binding=4) readonly buffer IncFace{ float incFace[]; };
layout(binding=5) buffer       CornNrm { float cn[]; };
uniform uint uTri;
uniform float uCos;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uTri) return;
    vec3 nf = vec3(faceN[3u*g], faceN[3u*g+1u], faceN[3u*g+2u]);
    for (int e = 0; e < 3; ++e) {
        uint v = idx[3u*g + uint(e)];
        uint cnt = incPos[v], off = incOff[v];
        vec3 acc = vec3(0.0);
        for (uint k = 0u; k < cnt; ++k) {
            vec3 g2 = vec3(incFace[3u*(off+k)], incFace[3u*(off+k)+1u], incFace[3u*(off+k)+2u]);
            if (dot(nf, g2) >= uCos) acc += g2;
        }
        float l = length(acc);
        vec3 res = (l > 1e-12) ? (acc / l) : nf;
        uint ci = 3u*g + uint(e);
        cn[3u*ci] = res.x; cn[3u*ci+1u] = res.y; cn[3u*ci+2u] = res.z;
    }
}
)";

// ---- clustering -----------------------------------------------------------

static const char* PROJ_COMMON = R"(
layout(binding=0) readonly buffer Pos { float pos[]; };
layout(binding=1) readonly buffer Idx { uint  idx[]; };
void proj_setup(uint g, uint axis, vec4 gp, out vec3 dd, out vec3 uu, out vec3 vv,
                out float pd[3], out vec2 pu, out vec2 pv, out float dmin, out float dmax){
    axis_basis(axis, dd, uu, vv);
    float dlo = 1e30, dhi = -1e30;
    float umin = 1e30, umax = -1e30, vmin = 1e30, vmax = -1e30;
    for (int e = 0; e < 3; ++e) {
        uint vi = idx[3u*g + uint(e)];
        vec3 p = vec3(pos[3u*vi], pos[3u*vi+1u], pos[3u*vi+2u]);
        float d = dot(p, dd) - gp.z;
        float u = (dot(p, uu) - gp.x) / gp.w;
        float v = (dot(p, vv) - gp.y) / gp.w;
        pd[e] = d;
        dlo = min(dlo, d); dhi = max(dhi, d);
        umin = min(umin, u); umax = max(umax, u);
        vmin = min(vmin, v); vmax = max(vmax, v);
    }
    dmin = dlo; dmax = dhi;
    pu = vec2(umin, umax); pv = vec2(vmin, vmax);
}
)";

static const char* K_CELLS = R"(
layout(binding=2) readonly buffer TriAxis { uint  axis[]; };
layout(binding=3) readonly buffer TriSub  { uint  sub[]; };
layout(binding=4) readonly buffer Grid    { vec4  grid[]; };
layout(binding=5) buffer       CellMin  { uint  cmin[]; };
layout(binding=6) buffer       CellMax  { uint  cmax[]; };
layout(binding=7) buffer       Isol     { uint  isol[]; };
uniform uint uTri; uniform uint uMaxFoot;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uTri) return;
    uint ax = axis[g];
    vec4 gp = grid[sub[g] * 6u + ax];
    vec3 dd, uu, vv; float pd[3]; vec2 pu, pv; float dmin, dmax;
    proj_setup(g, ax, gp, dd, uu, vv, pd, pu, pv, dmin, dmax);
    int c0 = int(floor(pu.x - 0.5)), c1 = int(ceil(pu.y + 0.5));
    int r0 = int(floor(pv.x - 0.5)), r1 = int(ceil(pv.y + 0.5));
    uint fw = uint(max(c1 - c0, 0)), fh = uint(max(r1 - r0, 0));
    if (fw == 0u || fh == 0u || fw > uMaxFoot || fh > uMaxFoot) { isol[g] = 1u; return; }
    uint base = (sub[g] * 6u + ax) * UCELL;
    uint dminb = f2s(dmin), dmaxb = f2s(dmax);
    for (uint fy = 0u; fy < fh; ++fy) {
        int cy = r0 + int(fy);
        if (cy < 0 || cy >= int(GRID)) continue;
        for (uint fx = 0u; fx < fw; ++fx) {
            int cx = c0 + int(fx);
            if (cx < 0 || cx >= int(GRID)) continue;
            uint cell = base + uint(cy) * GRID + uint(cx);
            atomicMin(cmin[cell], dminb);
            atomicMax(cmax[cell], dmaxb);
        }
    }
}
)";

static const char* K_CELLUNION = R"(
layout(binding=2) readonly buffer TriAxis { uint  axis[]; };
layout(binding=3) readonly buffer TriSub  { uint  sub[]; };
layout(binding=4) readonly buffer Grid    { vec4  grid[]; };
layout(binding=5) readonly buffer CellMin { uint  cmin[]; };
layout(binding=6) readonly buffer CellMax { uint  cmax[]; };
layout(binding=7) buffer       CellOwn  { uint  cown[]; };
layout(binding=8) buffer       Parent   { uint  parent[]; };
layout(binding=9) readonly buffer Isol   { uint  isol[]; };
uniform uint uTri; uniform uint uMaxFoot; uniform float uEps;
uint find_p(uint x){
    for (int it = 0; it < 48; ++it) {
        uint p = parent[x];
        if (p == x) break;
        x = p;
    }
    return x;
}
void un(uint a, uint b){
    for (int it = 0; it < 16; ++it) {
        uint ra = find_p(a), rb = find_p(b);
        if (ra == rb) return;
        if (ra > rb) { uint t = ra; ra = rb; rb = t; }
        uint old = atomicCompSwap(parent[rb], rb, ra);
        if (old == rb) return;
        b = old;
    }
}
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uTri) return;
    if (isol[g] == 1u) return;
    uint ax = axis[g];
    vec4 gp = grid[sub[g] * 6u + ax];
    vec3 dd, uu, vv; float pd[3]; vec2 pu, pv; float dmin, dmax;
    proj_setup(g, ax, gp, dd, uu, vv, pd, pu, pv, dmin, dmax);
    int c0 = int(floor(pu.x - 0.5)), c1 = int(ceil(pu.y + 0.5));
    int r0 = int(floor(pv.x - 0.5)), r1 = int(ceil(pv.y + 0.5));
    uint fw = uint(max(c1 - c0, 0)), fh = uint(max(r1 - r0, 0));
    if (fw == 0u || fh == 0u || fw > uMaxFoot || fh > uMaxFoot) return;
    uint base = (sub[g] * 6u + ax) * UCELL;
    for (uint fy = 0u; fy < fh; ++fy) {
        int cy = r0 + int(fy);
        if (cy < 0 || cy >= int(GRID)) continue;
        for (uint fx = 0u; fx < fw; ++fx) {
            int cx = c0 + int(fx);
            if (cx < 0 || cx >= int(GRID)) continue;
            uint cell = base + uint(cy) * GRID + uint(cx);
            uint mn = cmin[cell], mx = cmax[cell];
            if (mn == SENT || mx == 0u) continue;
            if (s2f(mx) - s2f(mn) <= uEps) {
                uint old = atomicCompSwap(cown[cell], SENT, g);
                if (old != SENT) un(g, old);
            }
        }
    }
}
)";

static const char* K_FIX = R"(
layout(binding=0) buffer Parent { uint parent[]; };
uniform uint uTri;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uTri) return;
    uint p1 = parent[g];
    if (p1 == g) return;
    uint p2 = parent[p1];
    if (p1 != p2) parent[g] = p2;
}
)";

static const char* K_SIZES = R"(
layout(binding=0) readonly buffer Parent    { uint parent[]; };
layout(binding=1) buffer       PatchSize { uint size[]; };
uniform uint uTri;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uTri) return;
    atomicAdd(size[parent[g]], 1u);
}
)";

static const char* K_MERGEA = R"(
layout(binding=2) readonly buffer TriAxis  { uint  axis[]; };
layout(binding=3) readonly buffer TriSub   { uint  sub[]; };
layout(binding=4) readonly buffer Grid     { vec4  grid[]; };
layout(binding=5) readonly buffer CellMin  { uint  cmin[]; };
layout(binding=6) readonly buffer CellMax  { uint  cmax[]; };
layout(binding=7) buffer       CellMerg  { uint  cmerg[]; };
layout(binding=8) readonly buffer Parent   { uint  parent[]; };
layout(binding=9) readonly buffer PatchSize{ uint  size[]; };
layout(binding=10) readonly buffer Isol    { uint  isol[]; };
uniform uint uTri; uniform uint uMaxFoot; uniform float uEps; uniform uint uMinPatch;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uTri) return;
    if (isol[g] == 1u) return;
    uint root = parent[g];
    if (size[root] < uMinPatch) return;
    uint ax = axis[g];
    vec4 gp = grid[sub[g] * 6u + ax];
    vec3 dd, uu, vv; float pd[3]; vec2 pu, pv; float dmin, dmax;
    proj_setup(g, ax, gp, dd, uu, vv, pd, pu, pv, dmin, dmax);
    int c0 = int(floor(pu.x - 0.5)), c1 = int(ceil(pu.y + 0.5));
    int r0 = int(floor(pv.x - 0.5)), r1 = int(ceil(pv.y + 0.5));
    uint fw = uint(max(c1 - c0, 0)), fh = uint(max(r1 - r0, 0));
    if (fw == 0u || fh == 0u || fw > uMaxFoot || fh > uMaxFoot) return;
    uint base = (sub[g] * 6u + ax) * UCELL;
    for (uint fy = 0u; fy < fh; ++fy) {
        int cy = r0 + int(fy);
        if (cy < 0 || cy >= int(GRID)) continue;
        for (uint fx = 0u; fx < fw; ++fx) {
            int cx = c0 + int(fx);
            if (cx < 0 || cx >= int(GRID)) continue;
            uint cell = base + uint(cy) * GRID + uint(cx);
            uint mn = cmin[cell], mx = cmax[cell];
            if (mn == SENT || mx == 0u) continue;
            if (s2f(mx) - s2f(mn) <= uEps)
                atomicCompSwap(cmerg[cell], SENT, root);
        }
    }
}
)";

static const char* K_MERGEB = R"(
layout(binding=2) readonly buffer TriAxis  { uint  axis[]; };
layout(binding=3) readonly buffer TriSub   { uint  sub[]; };
layout(binding=4) readonly buffer Grid     { vec4  grid[]; };
layout(binding=5) readonly buffer CellMin  { uint  cmin[]; };
layout(binding=6) readonly buffer CellMax  { uint  cmax[]; };
layout(binding=7) readonly buffer CellMerg { uint  cmerg[]; };
layout(binding=8) readonly buffer Parent   { uint  parent[]; };
layout(binding=9) readonly buffer PatchSize{ uint  size[]; };
layout(binding=10) buffer       Viol     { uint  viol[]; };
layout(binding=11) buffer       Best     { uint  best[]; };
layout(binding=12) readonly buffer Isol   { uint  isol[]; };
uniform uint uTri; uniform uint uMaxFoot; uniform float uEps; uniform uint uMinPatch;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uTri) return;
    if (isol[g] == 1u) return;
    uint root = parent[g];
    if (size[root] >= uMinPatch) return;
    uint ax = axis[g];
    vec4 gp = grid[sub[g] * 6u + ax];
    vec3 dd, uu, vv; float pd[3]; vec2 pu, pv; float dmin, dmax;
    proj_setup(g, ax, gp, dd, uu, vv, pd, pu, pv, dmin, dmax);
    int c0 = int(floor(pu.x - 0.5)), c1 = int(ceil(pu.y + 0.5));
    int r0 = int(floor(pv.x - 0.5)), r1 = int(ceil(pv.y + 0.5));
    uint fw = uint(max(c1 - c0, 0)), fh = uint(max(r1 - r0, 0));
    if (fw == 0u || fh == 0u || fw > uMaxFoot || fh > uMaxFoot) return;
    uint base = (sub[g] * 6u + ax) * UCELL;
    for (uint fy = 0u; fy < fh; ++fy) {
        int cy = r0 + int(fy);
        if (cy < 0 || cy >= int(GRID)) continue;
        for (uint fx = 0u; fx < fw; ++fx) {
            int cx = c0 + int(fx);
            if (cx < 0 || cx >= int(GRID)) continue;
            uint cell = base + uint(cy) * GRID + uint(cx);
            uint mn = cmin[cell], mx = cmax[cell];
            if (mn == SENT || mx == 0u) continue;
            if (s2f(mx) - s2f(mn) > uEps) continue;
            uint tgt = cmerg[cell];
            if (tgt == SENT || tgt == root) continue;
            float tmin = min(s2f(mn), dmin), tmax = max(s2f(mx), dmax);
            if (tmax - tmin > uEps) atomicMax(viol[root], 1u);
            else atomicMax(viol[root], 0u);
            uint key = (min(size[tgt], 4095u) << 20u) | tgt;
            atomicMax(best[root], key);
        }
    }
}
)";

static const char* K_MERGEC = R"(
layout(binding=0) readonly buffer PatchSize { uint size[]; };
layout(binding=1) readonly buffer Viol     { uint viol[]; };
layout(binding=2) readonly buffer Best     { uint best[]; };
layout(binding=3) buffer       Parent    { uint parent[]; };
uniform uint uTri; uniform uint uMinPatch;
uint find_p(uint x){
    for (int it = 0; it < 48; ++it) {
        uint p = parent[x];
        if (p == x) break;
        x = p;
    }
    return x;
}
void main(){
    uint root = gl_GlobalInvocationID.x;
    if (root >= uTri) return;
    if (size[root] >= uMinPatch) return;
    if (viol[root] != 0u) return;
    uint bk = best[root];
    if (bk == SENT) return;
    uint tgt = bk & 0xFFFFFu;
    if (tgt == root || size[tgt] <= size[root]) return;
    if (find_p(root) != root || find_p(tgt) != tgt) return;
    uint a = root, b = tgt;
    if (a > b) { uint t = a; a = b; b = t; }
    atomicCompSwap(parent[b], b, a);
}
)";

// ---- patch finalize --------------------------------------------------------

static const char* K_FINALIZE = R"(
layout(binding=0) readonly buffer Parent    { uint parent[]; };
layout(binding=1) readonly buffer Pos       { float pos[]; };
layout(binding=2) readonly buffer Idx       { uint  idx[]; };
layout(binding=3) readonly buffer TriAxis   { uint  axis[]; };
layout(binding=4) buffer       PatchAxis  { uint  paxis[]; };
layout(binding=5) buffer       PMin       { uint  pmin[]; };
layout(binding=6) buffer       PMax       { uint  pmax[]; };
uniform uint uTri;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uTri) return;
    uint r = parent[g];
    paxis[r] = axis[g];
    for (int e = 0; e < 3; ++e) {
        uint vi = idx[3u*g + uint(e)];
        for (int k = 0; k < 3; ++k) {
            float v = pos[3u*vi + uint(k)];
            atomicMin(pmin[3u*r + uint(k)], f2s(v));
            atomicMax(pmax[3u*r + uint(k)], f2s(v));
        }
    }
}
)";

static const char* K_OCC = R"(
layout(binding=0) readonly buffer CellMin { uint cmin[]; };
layout(binding=1) readonly buffer CellMax { uint cmax[]; };
layout(binding=2) buffer       Occ      { uint occ[1]; };
uniform uint uCells; uniform float uEps;
void main(){
    uint i = gl_GlobalInvocationID.x;
    if (i >= uCells) return;
    uint mn = cmin[i], mx = cmax[i];
    if (mn == SENT || mx == 0u) return;
    float over = s2f(mx) - s2f(mn) - uEps;
    if (over > 0.0) atomicMax(occ[0], f2s(over));
}
)";

// ---- triangle reorder ------------------------------------------------------

static const char* K_REORDER_COUNT = R"(
layout(binding=0) readonly buffer TriLabel   { uint label[]; };
layout(binding=1) buffer       PatchTriCnt { uint cnt[]; };
uniform uint uTri;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uTri) return;
    atomicAdd(cnt[label[g]], 1u);
}
)";

static const char* K_REORDER_FILL = R"(
layout(binding=0) readonly buffer TriLabel   { uint label[]; };
layout(binding=1) buffer       PatchTriOff { uint off[]; };
layout(binding=2) buffer       TriSorted   { uint sorted[]; };
uniform uint uTri;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uTri) return;
    uint slot = atomicAdd(off[label[g]], 1u);
    sorted[slot] = g;
}
)";

// ---- rasterisation ---------------------------------------------------------

static const char* K_RASTER = R"(
layout(binding=0) readonly buffer Pos      { float pos[]; };
layout(binding=1) readonly buffer Idx      { uint  idx[]; };
layout(binding=2) readonly buffer UV       { float uv[]; };
layout(binding=3) readonly buffer CornNrm  { float cn[]; };
layout(binding=4) readonly buffer TriLabel { uint  label[]; };
layout(binding=5) readonly buffer PatchAxis{ uint  paxis[]; };
layout(binding=6) readonly buffer PABB     { float pabb[]; };
layout(binding=7) readonly buffer Rect     { uint  rect[]; };
layout(binding=8) readonly buffer NTab     { uint  ntab[]; };
layout(binding=9)  buffer A_Dmin  { uint  dmin[]; };
layout(binding=10) buffer A_Dmax  { uint  dmax[]; };
layout(binding=11) buffer A_Thick { uint  thick[]; };
layout(binding=12) buffer A_Uv    { float auv[]; };
layout(binding=13) buffer A_Nrm   { float anrm[]; };
uniform uint uTri; uniform uint uW;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uTri) return;
    uint L = label[g];
    uint ax = paxis[L];
    vec3 dd, uu, vv; axis_basis(ax, dd, uu, vv);
    uint ax0 = rect[4u*L], ay0 = rect[4u*L+1u], tw = rect[4u*L+2u], th = rect[4u*L+3u];
    if (tw == 0u || th == 0u) return;
    float mn0 = pabb[6u*L], mn1 = pabb[6u*L+1u], mn2 = pabb[6u*L+2u];
    float mx0 = pabb[6u*L+3u], mx1 = pabb[6u*L+4u], mx2 = pabb[6u*L+5u];
    float uLo = proj_min3(mn0,mn1,mn2,mx0,mx1,mx2,uu), uHi = proj_max3(mn0,mn1,mn2,mx0,mx1,mx2,uu);
    float vLo = proj_min3(mn0,mn1,mn2,mx0,mx1,mx2,vv), vHi = proj_max3(mn0,mn1,mn2,mx0,mx1,mx2,vv);
    float pLo = proj_min3(mn0,mn1,mn2,mx0,mx1,mx2,dd);
    float tu = max(uHi - uLo, 1e-9) / float(tw), tv = max(vHi - vLo, 1e-9) / float(th);
    vec3 p[3]; vec2 puv[3]; float pd[3]; vec2 vuv[3]; vec3 nrm[3];
    float tdmin = 1e30, tdmax = -1e30;
    for (int e = 0; e < 3; ++e) {
        uint vi = idx[3u*g + uint(e)];
        vec3 pp = vec3(pos[3u*vi], pos[3u*vi+1u], pos[3u*vi+2u]);
        p[e] = pp;
        puv[e] = vec2((dot(pp, uu) - uLo) / tu, (dot(pp, vv) - vLo) / tv);
        pd[e] = dot(pp, dd) - pLo;
        tdmin = min(tdmin, pd[e]); tdmax = max(tdmax, pd[e]);
        vuv[e] = vec2(uv[2u*vi], uv[2u*vi+1u]);
        uint ci = 3u*g + uint(e);
        nrm[e] = vec3(cn[3u*ci], cn[3u*ci+1u], cn[3u*ci+2u]);
    }
    float thickTri = tdmax - tdmin;
    int c0 = int(floor(min(min(puv[0].x,puv[1].x),puv[2].x) - 0.5));
    int c1 = int(ceil(max(max(puv[0].x,puv[1].x),puv[2].x) + 0.5));
    int r0 = int(floor(min(min(puv[0].y,puv[1].y),puv[2].y) - 0.5));
    int r1 = int(ceil(max(max(puv[0].y,puv[1].y),puv[2].y) + 0.5));
    c0 = max(c0, 0); r0 = max(r0, 0);
    c1 = min(c1, int(tw)); r1 = min(r1, int(th));
    if (c1 - c0 <= 0 || r1 - r0 <= 0) return;
    vec2 v0 = puv[1] - puv[0], v1 = puv[2] - puv[0];
    float den = v0.x * v1.y - v1.x * v0.y;
    for (int yy = r0; yy < r1; ++yy) {
        for (int xx = c0; xx < c1; ++xx) {
            vec2 tp = vec2((float(xx) + 0.5) * tu, (float(yy) + 0.5) * tv);
            vec2 v2 = tp - puv[0];
            if (abs(den) < 1e-12) continue;
            float l2 = (v2.x * v1.y - v1.x * v2.y) / den;
            float l1 = (v0.x * v2.y - v2.x * v0.y) / den;
            float l0 = 1.0 - l1 - l2;
            if (l0 < -0.02 || l1 < -0.02 || l2 < -0.02) continue;
            float dpt = l0 * pd[0] + l1 * pd[1] + l2 * pd[2];
            vec2 vv2 = l0 * vuv[0] + l1 * vuv[1] + l2 * vuv[2];
            vec3 nn = normalize(l0 * nrm[0] + l1 * nrm[1] + l2 * nrm[2]);
            vec2 oct = encOct(nn);
            uint cell = (ay0 + uint(yy)) * uW + (ax0 + uint(xx));
            atomicMin(dmin[cell], f2s(dpt));
            atomicMax(dmax[cell], f2s(dpt));
            atomicMax(thick[cell], f2s(thickTri));
            auv[2u*cell] = vv2.x; auv[2u*cell+1u] = vv2.y;
            anrm[2u*cell] = oct.x; anrm[2u*cell+1u] = oct.y;
        }
    }
}
)";

// ---- hole filling ----------------------------------------------------------

static const char* K_HF_SETUP = R"(
layout(binding=0) readonly buffer Rect { uint rect[]; };
layout(binding=1) buffer ARect { uint arect[]; };
uniform uint uPatches; uniform uint uW;
void main(){
    uint p = gl_GlobalInvocationID.x;
    if (p >= uPatches) return;
    uint ax0 = rect[4u*p], ay0 = rect[4u*p+1u], tw = rect[4u*p+2u], th = rect[4u*p+3u];
    for (uint y = 0u; y < th; ++y)
        for (uint x = 0u; x < tw; ++x)
            arect[(ay0 + y) * uW + (ax0 + x)] = p;
}
)";

static const char* K_HF_BORDER = R"(
layout(binding=0) readonly buffer Rect    { uint rect[]; };
layout(binding=1) readonly buffer A_Dmin  { uint dmin[]; };
layout(binding=2) buffer AState { uint st[]; };
uniform uint uPatches; uniform uint uW;
void main(){
    uint p = gl_GlobalInvocationID.x;
    if (p >= uPatches) return;
    uint ax0 = rect[4u*p], ay0 = rect[4u*p+1u], tw = rect[4u*p+2u], th = rect[4u*p+3u];
    for (uint x = 0u; x < tw; ++x) {
        uint c0 = (ay0) * uW + (ax0 + x);
        uint c1 = (ay0 + th - 1u) * uW + (ax0 + x);
        if (dmin[c0] == SENT) st[c0] = 1u;
        if (dmin[c1] == SENT) st[c1] = 1u;
    }
    for (uint y = 1u; y + 1u < th; ++y) {
        uint c0 = (ay0 + y) * uW + (ax0);
        uint c1 = (ay0 + y) * uW + (ax0 + tw - 1u);
        if (dmin[c0] == SENT) st[c0] = 1u;
        if (dmin[c1] == SENT) st[c1] = 1u;
    }
}
)";

static const char* K_HF_FLOOD = R"(
layout(binding=0) readonly buffer Rect { uint rect[]; };
layout(binding=1) readonly buffer ARect { uint arect[]; };
layout(binding=2) readonly buffer A_Dmin { uint dmin[]; };
layout(binding=3) buffer AState { uint st[]; };
layout(binding=4) buffer Changed { uint changed[1]; };
uniform uint uW; uniform uint uH;
void main(){
    uint gid = gl_GlobalInvocationID.x;
    uint total = uW * uH;
    if (gid >= total) return;
    uint p = arect[gid];
    if (p == SENT) return;
    if (st[gid] == 1u || dmin[gid] != SENT) return;
    uint ax0 = rect[4u*p], ay0 = rect[4u*p+1u], tw = rect[4u*p+2u], th = rect[4u*p+3u];
    uint x = gid % uW, y = gid / uW;
    bool hit = false;
    if (x + 1u < ax0 + tw && st[gid + 1u] == 1u) hit = true;
    if (x > ax0 && st[gid - 1u] == 1u) hit = true;
    if (y + 1u < ay0 + th && st[gid + uW] == 1u) hit = true;
    if (y > ay0 && st[gid - uW] == 1u) hit = true;
    if (hit) { st[gid] = 1u; atomicMax(changed[0], 1u); }
}
)";

static const char* K_HF_FILL = R"(
layout(binding=0) readonly buffer Rect  { uint rect[]; };
layout(binding=1) readonly buffer ARect { uint arect[]; };
layout(binding=2) buffer A_Dmin  { uint dmin[]; };
layout(binding=3) buffer A_Dmax  { uint dmax[]; };
layout(binding=4) buffer A_Thick { uint thick[]; };
layout(binding=5) buffer A_Uv    { float auv[]; };
layout(binding=6) buffer A_Nrm   { float anrm[]; };
layout(binding=7) buffer AState  { uint st[]; };
layout(binding=8) buffer Changed { uint changed[1]; };
uniform uint uW; uniform uint uH;
void main(){
    uint gid = gl_GlobalInvocationID.x;
    uint total = uW * uH;
    if (gid >= total) return;
    uint p = arect[gid];
    if (p == SENT) return;
    if (st[gid] != 0u || dmin[gid] != SENT) return;
    uint ax0 = rect[4u*p], ay0 = rect[4u*p+1u], tw = rect[4u*p+2u], th = rect[4u*p+3u];
    uint x = gid % uW, y = gid / uW;
    uint src = SENT;
    if (x + 1u < ax0 + tw && dmin[gid + 1u] != SENT) src = gid + 1u;
    else if (x > ax0 && dmin[gid - 1u] != SENT) src = gid - 1u;
    else if (y + 1u < ay0 + th && dmin[gid + uW] != SENT) src = gid + uW;
    else if (y > ay0 && dmin[gid - uW] != SENT) src = gid - uW;
    if (src == SENT) return;
    dmin[gid] = dmin[src];
    dmax[gid] = dmax[src];
    thick[gid] = thick[src];
    auv[2u*gid] = auv[2u*src]; auv[2u*gid+1u] = auv[2u*src+1u];
    anrm[2u*gid] = anrm[2u*src]; anrm[2u*gid+1u] = anrm[2u*src+1u];
    st[gid] = 2u;
    atomicMax(changed[0], 1u);
}
)";

// ---- mip chains ------------------------------------------------------------

static const char* K_RANGE = R"(
layout(binding=0) readonly buffer A_Dmin { uint dmin[]; };
layout(binding=1) readonly buffer A_Dmax { uint dmax[]; };
layout(binding=2) readonly buffer A_Thick { uint thick[]; };
layout(binding=3) readonly buffer A_Uv { float auv[]; };
layout(binding=4) buffer Range { uint rg[8]; };
uniform uint uW; uniform uint uH;
void main(){
    uint gid = gl_GlobalInvocationID.x;
    uint total = uW * uH;
    if (gid >= total) return;
    if (dmin[gid] == SENT) return;
    atomicMin(rg[0], dmin[gid]);      // depth min
    atomicMax(rg[1], dmax[gid]);      // depth max
    atomicMax(rg[2], thick[gid]);     // thick max
    atomicMin(rg[3], f2s(auv[2u*gid]));   // u min
    atomicMax(rg[4], f2s(auv[2u*gid]));   // u max
    atomicMin(rg[5], f2s(auv[2u*gid+1u])); // v min
    atomicMax(rg[6], f2s(auv[2u*gid+1u])); // v max
}
)";

static const char* K_MIPAGG = R"(
layout(binding=0) readonly buffer Node0 { uint n0[]; };
layout(binding=1) readonly buffer Node1 { uint n1[]; };
layout(binding=2) readonly buffer Rect  { uint rect[]; };
layout(binding=3) readonly buffer NTab  { uint ntab[]; };
layout(binding=4) readonly buffer A_Dmin { uint dmin[]; };
layout(binding=5) readonly buffer A_Dmax { uint dmax[]; };
layout(binding=6) readonly buffer A_Thick { uint thick[]; };
layout(binding=7) readonly buffer A_Uv  { float auv[]; };
layout(binding=8) readonly buffer A_Nrm { float anrm[]; };
layout(binding=9)  buffer MipVal { uint mv[]; };
layout(binding=10) buffer MipFlag { uint mflag[]; };
layout(binding=11) buffer MipNext { uint mnext[1]; };
layout(binding=12) buffer Node0B { uint nb0[]; };
layout(binding=13) buffer Node1B { uint nb1[]; };
uniform uint uLevel; uniform uint uNodeCount; uniform uint uValBase; uniform uint uW;
uniform uint uLeaf;
uniform float uDepthTol; uniform float uThickTol; uniform float uUvTol;
uniform float uDmin, uDmax, uThickMax, uUmin, uUmax, uVmin, uVmax;
void main(){
    uint i = gl_GlobalInvocationID.x;
    if (i >= uNodeCount) return;
    uint pa = n0[i];
    uint xy = n1[i];
    uint x = xy >> 16u, y = xy & 0xFFFFu;
    uint tw = rect[4u*pa+2u], th = rect[4u*pa+3u];
    uint s = ntab[pa] >> uLevel;
    if (s == 0u) { mflag[i] = 0u; return; }
    uint ax0 = rect[4u*pa], ay0 = rect[4u*pa+1u];
    uint cnt = 0u;
    uint dmn = SENT, dmx = 0u, thk = 0u;
    float usum = 0.0, vsum = 0.0;
    vec3 nsum = vec3(0.0);
    uint umn = SENT, umx = 0u, vmn = SENT, vmx = 0u;
    for (uint j = 0u; j < s; ++j) {
        uint y0 = ay0 + y * s + j;
        if (y0 >= ay0 + th) continue;
        for (uint ii = 0u; ii < s; ++ii) {
            uint x0 = ax0 + x * s + ii;
            if (x0 >= ax0 + tw) continue;
            uint c = y0 * uW + x0;
            uint dm = dmin[c];
            if (dm == SENT) continue;
            cnt++;
            dmn = min(dmn, dm); dmx = max(dmx, dm);
            thk = max(thk, thick[c]);
            umn = min(umn, f2s(auv[2u*c])); umx = max(umx, f2s(auv[2u*c]));
            vmn = min(vmn, f2s(auv[2u*c+1u])); vmx = max(vmx, f2s(auv[2u*c+1u]));
            usum += auv[2u*c]; vsum += auv[2u*c+1u];
            nsum += decOct(vec2(anrm[2u*c], anrm[2u*c+1u]));
        }
    }
    bool has = cnt > 0u;
    bool partial = has && cnt < s * s;
    float dlo = has ? s2f(dmn) : 0.0, dhi = has ? s2f(dmx) : 0.0;
    bool exceeds = (dhi - dlo) > uDepthTol || (has && s2f(thk) > uThickTol);
    if (umn != SENT) {
        float ur = s2f(umx) - s2f(umn), vr = s2f(vmx) - s2f(vmn);
        if (ur > uUvTol || vr > uUvTol) exceeds = true;
    }
    bool sub = has && s > uLeaf && uLevel + 1u < 12u && (exceeds || partial);
    uint dmnq = 0u, dmxq = 0u, thkq = 0u, uq = 0u, vq = 0u, nxq = 0u, nyq = 0u;
    if (has) {
        dmnq = uint(round((dlo - uDmin) / max(uDmax - uDmin, 1e-9) * 255.0));
        dmxq = uint(round((dhi - uDmin) / max(uDmax - uDmin, 1e-9) * 255.0));
        thkq = uint(round(min(1.0, s2f(thk) / max(uThickMax, 1e-9)) * 255.0));
        uq = uint(round((usum / float(cnt) - uUmin) / max(uUmax - uUmin, 1e-9) * 255.0));
        vq = uint(round((vsum / float(cnt) - uVmin) / max(uVmax - uVmin, 1e-9) * 255.0));
        vec3 nn = normalize(nsum);
        vec2 oct = encOct(nn);
        nxq = uint(round(oct.x * 255.0)); nyq = uint(round(oct.y * 255.0));
    }
    mv[2u*uValBase + 2u*i] = (dmnq << 24u) | (dmxq << 16u) | (thkq << 8u) | uq;
    mv[2u*uValBase + 2u*i + 1u] = (vq << 24u) | (nxq << 16u) | (nyq << 8u);
    mflag[i] = sub ? 1u : 0u;
    if (sub) {
        uint off = atomicAdd(mnext[0], 1u);
        uint b = off * 4u;
        nb0[b] = pa;   nb1[b] = (2u*x) << 16u | (2u*y);
        nb0[b+1u] = pa; nb1[b+1u] = (2u*x + 1u) << 16u | (2u*y);
        nb0[b+2u] = pa; nb1[b+2u] = (2u*x) << 16u | (2u*y + 1u);
        nb0[b+3u] = pa; nb1[b+3u] = (2u*x + 1u) << 16u | (2u*y + 1u);
    }
}
)";

// ---- BVH -------------------------------------------------------------------

static const char* K_MORTON = R"(
layout(binding=0) readonly buffer PABB { float pabb[]; };
layout(binding=1) readonly buffer ModelBB { float mbb[]; };
layout(binding=2) buffer Morton { uint morton[]; };
uniform uint uPatches;
uint spread10(uint x){
    x &= 0x3FFu;
    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x << 8))  & 0x0300F00Fu;
    x = (x | (x << 4))  & 0x030C30C3u;
    x = (x | (x << 2))  & 0x09249249u;
    return x;
}
float gridf(float v, float lo, float hi){
    if (hi <= lo) return 0.5;
    return clamp((v - lo) / (hi - lo), 0.0, 1.0);
}
void main(){
    uint p = gl_GlobalInvocationID.x;
    if (p >= uPatches) return;
    float mx = (pabb[6u*p] + pabb[6u*p+3u]) * 0.5;
    float my = (pabb[6u*p+1u] + pabb[6u*p+4u]) * 0.5;
    float mz = (pabb[6u*p+2u] + pabb[6u*p+5u]) * 0.5;
    uint ix = uint(gridf(mx, mbb[0], mbb[3]) * 1023.0);
    uint iy = uint(gridf(my, mbb[1], mbb[4]) * 1023.0);
    uint iz = uint(gridf(mz, mbb[2], mbb[5]) * 1023.0);
    morton[p] = spread10(ix) | (spread10(iy) << 1u) | (spread10(iz) << 2u);
}
)";

static const char* K_RADIX_COUNT = R"(
layout(binding=0) readonly buffer KeyIn { uint kin[]; };
layout(binding=1) buffer Counts { uint cnt[256]; };
uniform uint uN; uniform uint uPass;
shared uint hist[256];
void main(){
    uint i = gl_GlobalInvocationID.x;
    hist[gl_LocalInvocationID.x] = 0u;
    barrier();
    if (i < uN) {
        uint d = (kin[i] >> (8u * uPass)) & 255u;
        atomicAdd(hist[d], 1u);
    }
    barrier();
    atomicAdd(cnt[gl_LocalInvocationID.x], hist[gl_LocalInvocationID.x]);
}
)";

static const char* K_RADIX_SCAN = R"(
layout(binding=0) buffer Counts { uint cnt[256]; };
shared uint s[256];
shared uint orig[256];
void main(){
    uint i = gl_LocalInvocationID.x;
    s[i] = cnt[i];
    orig[i] = cnt[i];
    barrier();
    for (uint d = 1u; d < 256u; d <<= 1u) {
        uint v = s[i];
        if (i >= d) v += s[i - d];
        barrier();
        s[i] = v;
        barrier();
    }
    cnt[i] = s[i] - orig[i];
}
)";

static const char* K_RADIX_SCATTER = R"(
layout(binding=0) readonly buffer KeyIn  { uint kin[]; };
layout(binding=1) readonly buffer AuxIn  { uint ain[]; };
layout(binding=2) buffer KeyOut { uint kout[]; };
layout(binding=3) buffer AuxOut { uint aout[]; };
layout(binding=4) buffer Counts { uint cnt[256]; };
uniform uint uN; uniform uint uPass;
void main(){
    uint i = gl_GlobalInvocationID.x;
    if (i >= uN) return;
    uint k = kin[i];
    uint d = (k >> (8u * uPass)) & 255u;
    uint pos = atomicAdd(cnt[d], 1u);
    kout[pos] = k;
    aout[pos] = ain[i];
}
)";

static const char* K_BVSPLIT = R"(
layout(binding=0) buffer Range   { int range[]; };
layout(binding=1) buffer Child   { int child[]; };
layout(binding=2) buffer NodeCnt { uint nodeCnt[1]; };
layout(binding=3) readonly buffer LevelA { uint la[]; };
layout(binding=4) buffer LevelB { uint lb[]; };
layout(binding=5) buffer NextCnt { uint nc[1]; };
uniform uint uLevelCount;
void main(){
    uint i = gl_GlobalInvocationID.x;
    if (i >= uLevelCount) return;
    uint node = la[i];
    int rl = range[2*node], rr = range[2*node+1];
    if (rr <= rl) return;
    int mid = (rl + rr) >> 1;
    uint c = atomicAdd(nodeCnt[0], 2u);
    child[2*node] = int(c);
    child[2*node+1] = int(c + 1u);
    range[2*int(c)] = rl; range[2*int(c)+1] = mid;
    range[2*int(c+1u)] = mid + 1; range[2*int(c+1u)+1] = rr;
    uint off = atomicAdd(nc[0], 2u);
    lb[off] = c; lb[off + 1u] = c + 1u;
}
)";

static const char* K_BVAABB = R"(
layout(binding=0) readonly buffer Level { uint la[]; };
layout(binding=1) readonly buffer Range { int range[]; };
layout(binding=2) readonly buffer Child { int child[]; };
layout(binding=3) readonly buffer PABB  { float pabb[]; };
layout(binding=4) readonly buffer SortedP { uint sp[]; };
layout(binding=5) buffer AABB { float aabb[]; };
layout(binding=6) buffer Size { uint size[]; };
uniform uint uLevelCount;
void main(){
    uint i = gl_GlobalInvocationID.x;
    if (i >= uLevelCount) return;
    uint node = la[i];
    int rl = range[2*node], rr = range[2*node+1];
    if (rr <= rl) {
        uint p = sp[rl];
        aabb[6u*node] = pabb[6u*p];
        aabb[6u*node+1u] = pabb[6u*p+1u];
        aabb[6u*node+2u] = pabb[6u*p+2u];
        aabb[6u*node+3u] = pabb[6u*p+3u];
        aabb[6u*node+4u] = pabb[6u*p+4u];
        aabb[6u*node+5u] = pabb[6u*p+5u];
        size[node] = 1u;
        return;
    }
    int c0 = child[2*node], c1 = child[2*node+1];
    aabb[6u*node] = min(aabb[6u*c0], aabb[6u*c1]);
    aabb[6u*node+1u] = min(aabb[6u*c0+1u], aabb[6u*c1+1u]);
    aabb[6u*node+2u] = min(aabb[6u*c0+2u], aabb[6u*c1+2u]);
    aabb[6u*node+3u] = max(aabb[6u*c0+3u], aabb[6u*c1+3u]);
    aabb[6u*node+4u] = max(aabb[6u*c0+4u], aabb[6u*c1+4u]);
    aabb[6u*node+5u] = max(aabb[6u*c0+5u], aabb[6u*c1+5u]);
    size[node] = size[c0] + size[c1] + 1u;
}
)";

static const char* K_BVRENUMBER = R"(
layout(binding=0) readonly buffer Range { int range[]; };
layout(binding=1) readonly buffer Child { int child[]; };
layout(binding=2) readonly buffer SortedP { uint sp[]; };
layout(binding=3) readonly buffer AABB  { float aabb[]; };
layout(binding=4) readonly buffer Size  { uint size[]; };
layout(binding=5) buffer OutA { float outA[]; };
layout(binding=6) buffer OutB { int outB[]; };
uniform uint uPatches;
void main(){
    if (gl_GlobalInvocationID.x != 0u) return;
    int stack[128];
    int top = 0;
    stack[0] = 0;
    uint newId = 0u;
    while (top >= 0) {
        int n = stack[top];
        --top;
        uint oid = newId;
        ++newId;
        int rl = range[2*n], rr = range[2*n+1];
        int rightOff = 0, leafP = -1;
        if (rr <= rl) {
            leafP = int(sp[rl]);
        } else {
            int c0 = child[2*n], c1 = child[2*n+1];
            rightOff = int(size[c1]) + 1;
            stack[++top] = c1;
            stack[++top] = c0;
        }
        for (int k = 0; k < 6; ++k)
            outA[6u*oid + uint(k)] = aabb[6u*uint(n) + uint(k)];
        outB[2u*oid] = rightOff;
        outB[2u*oid+1u] = leafP;
    }
}
)";

// ======================================================================
// host pipeline state
// ======================================================================

struct GpuResult {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;         // smooth vertex normals (from crease corners)
    std::vector<uint32_t>  render_indices;  // patch-major index buffer
    std::vector<glm::vec3> patches_center;
    std::vector<uint32_t>  patch_tri_counts;
    std::vector<uint32_t>  patch_tri_offsets;
    uint32_t patch_count = 0;
    uint32_t tri_count = 0;
    int      atlas_w = 0, atlas_h = 0;
    uint64_t covered_texels = 0;
    uint64_t atlas_texels = 0;
    std::vector<uint32_t> mip_level_counts;
    uint32_t bvh_nodes = 0;
    float    occ_max_over = 0.0f;   // max (cell depth range - epsilon); <=0 => all occlusion-free
    std::vector<uint32_t> per_axis;
    std::unordered_map<std::string, double> timings;
};

struct Pipe {
    uint32_t V = 0, T = 0, S = 0, P = 0;
    uint32_t atlas_w = 0, atlas_h = 0;
    std::unordered_map<std::string, double> timings;

    Kern k_face, k_triprep, k_inccount, k_incfill, k_crease,
         k_cells, k_cellunion, k_fix, k_sizes, k_mergea, k_mergeb, k_mergec,
         k_finalize, k_occ, k_reorderc, k_reorderf, k_raster,
         k_hf_setup, k_hf_border, k_hf_flood, k_hf_fill,
         k_range, k_mipagg, k_morton, k_radix_count, k_radix_scan, k_radix_scatter,
         k_bvsplit, k_bvaabb, k_bvrenumber;

    gl::Buffer b_pos, b_idx, b_uv, b_triSub, b_faceN,
               b_incPos, b_incOff, b_incFill, b_incFace, b_cornNrm,
               b_subAABB, b_grid, b_triAxis,
               b_cellMin, b_cellMax, b_cellOwn, b_cellMerg,
               b_parent, b_isol, b_patchSize, b_viol, b_best,
               b_pAABBmin, b_pAABBmax, b_patchAxis,
               b_triLabel, b_patchAABB, b_patchRect, b_patchN,
               b_patchTriCnt, b_patchTriOff, b_triSorted,
               b_atlasDmin, b_atlasDmax, b_atlasThick, b_atlasUv, b_atlasNrm,
               b_atlasRect, b_atlasState, b_changed,
               b_occ,
               b_rk0, b_rk1, b_ra0, b_ra1, b_rcounts,
               b_morton, b_sortedP, b_mbb,
               b_bvRange, b_bvChild, b_bvNodeCnt, b_bvLevelA, b_bvLevelB,
               b_bvNextCnt, b_bvAABB, b_bvSize, b_bvOutA, b_bvOutB,
               b_mipN0, b_mipN1, b_mipN0b, b_mipN1b, b_mipVal, b_mipFlag,
               b_mipNext, b_mipRange;

    Pipe() {
        k_face      = make_kern(kern(K_FACE));
        k_triprep   = make_kern(kern(K_TRIPREP));
        k_inccount  = make_kern(kern(K_INCCOUNT));
        k_incfill   = make_kern(kern(K_INCFILL));
        k_crease    = make_kern(kern(K_CREASE));
        k_cells     = make_kern(kern(PROJ_COMMON) + K_CELLS);
        k_cellunion = make_kern(kern(PROJ_COMMON) + K_CELLUNION);
        k_fix       = make_kern(kern(K_FIX));
        k_sizes     = make_kern(kern(K_SIZES));
        k_mergea    = make_kern(kern(PROJ_COMMON) + K_MERGEA);
        k_mergeb    = make_kern(kern(PROJ_COMMON) + K_MERGEB);
        k_mergec    = make_kern(kern(K_MERGEC));
        k_finalize  = make_kern(kern(K_FINALIZE));
        k_occ       = make_kern(kern(K_OCC));
        k_reorderc  = make_kern(kern(K_REORDER_COUNT));
        k_reorderf  = make_kern(kern(K_REORDER_FILL));
        k_raster    = make_kern(kern(K_RASTER));
        k_hf_setup  = make_kern(kern(K_HF_SETUP));
        k_hf_border = make_kern(kern(K_HF_BORDER));
        k_hf_flood  = make_kern(kern(K_HF_FLOOD));
        k_hf_fill   = make_kern(kern(K_HF_FILL));
        k_range     = make_kern(kern(K_RANGE));
        k_mipagg    = make_kern(kern(K_MIPAGG));
        k_morton    = make_kern(kern(K_MORTON));
        k_radix_count   = make_kern(kern(K_RADIX_COUNT));
        k_radix_scan    = make_kern(kern(K_RADIX_SCAN));
        k_radix_scatter = make_kern(kern(K_RADIX_SCATTER));
        k_bvsplit    = make_kern(kern(K_BVSPLIT));
        k_bvaabb     = make_kern(kern(K_BVAABB));
        k_bvrenumber = make_kern(kern(K_BVRENUMBER));
    }
};

// ======================================================================
// CPU: weld + pack (the pieces left on the host)
// ======================================================================

static void weld_positions(const std::vector<gfx::Vertex>& vin,
                           std::vector<glm::vec3>& pos,
                           std::vector<glm::vec3>& nrm,
                           std::vector<glm::vec2>& uv,
                           std::vector<uint32_t>& idx,
                           float quant) {
    struct KV { int64_t kx, ky, kz; uint32_t src; };
    std::vector<KV> keys(vin.size());
    for (uint32_t i = 0; i < vin.size(); ++i) {
        keys[i] = {
            int64_t(std::llround(double(vin[i].position[0]) / quant)),
            int64_t(std::llround(double(vin[i].position[1]) / quant)),
            int64_t(std::llround(double(vin[i].position[2]) / quant)),
            i
        };
    }
    std::sort(keys.begin(), keys.end(), [](const KV& a, const KV& b) {
        if (a.kx != b.kx) return a.kx < b.kx;
        if (a.ky != b.ky) return a.ky < b.ky;
        if (a.kz != b.kz) return a.kz < b.kz;
        return a.src < b.src;
    });
    std::vector<uint32_t> remap(vin.size());
    std::vector<uint32_t> canon;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i == 0 || keys[i].kx != keys[i-1].kx || keys[i].ky != keys[i-1].ky ||
            keys[i].kz != keys[i-1].kz) {
            canon.push_back(keys[i].src);
        }
        remap[keys[i].src] = uint32_t(canon.size() - 1);
    }
    pos.reserve(canon.size()); nrm.reserve(canon.size()); uv.reserve(canon.size());
    for (uint32_t c : canon) {
        const auto& v = vin[c];
        pos.push_back(glm::vec3(v.position[0], v.position[1], v.position[2]));
        nrm.push_back(glm::vec3(v.normal[0], v.normal[1], v.normal[2]));
        uv.push_back(glm::vec2(v.texcoord[0], v.texcoord[1]));
    }
    for (auto& i : idx) i = remap[i];
}

struct PatchCpu {
    glm::vec3 aabb_min, aabb_max;
    int   axis = 0;
    float proj_u_lo = 0, proj_v_lo = 0, proj_u_size = 0, proj_v_size = 0;
    int   tex_w = 0, tex_h = 0;
    int   atlas_x = 0, atlas_y = 0;
    uint32_t tri_offset = 0, tri_count = 0;
};

static void size_textures(std::vector<PatchCpu>& patches, float density,
                          int min_tex, int max_tex) {
    for (auto& p : patches) {
        p.tex_w = std::clamp(int(std::ceil(p.proj_u_size * density)), min_tex, max_tex);
        p.tex_h = std::clamp(int(std::ceil(p.proj_v_size * density)), min_tex, max_tex);
    }
}

static float fit_to_budget(std::vector<PatchCpu>& patches, float base_density,
                           float budget, int min_tex, int max_tex) {
    float total = 0;
    for (auto& p : patches) total += float(p.tex_w) * p.tex_h;
    if (total <= budget) return base_density;
    float lo = 0.001f, hi = 1.0f;
    for (int it = 0; it < 32; ++it) {
        float mid = (lo + hi) * 0.5f;
        float t = 0;
        for (auto& p : patches)
            t += float(std::clamp(int(std::ceil(p.proj_u_size * base_density * mid)), min_tex, max_tex)) *
                 float(std::clamp(int(std::ceil(p.proj_v_size * base_density * mid)), min_tex, max_tex));
        if (t > budget) hi = mid; else lo = mid;
    }
    float final_density = base_density * lo;
    size_textures(patches, final_density, min_tex, max_tex);
    return final_density;
}

static void pack_atlas(std::vector<PatchCpu>& patches, int& W, int& H) {
    std::sort(patches.begin(), patches.end(), [](const PatchCpu& a, const PatchCpu& b) {
        return std::max(a.tex_w, a.tex_h) > std::max(b.tex_w, b.tex_h);
    });
    W = 1024;
    constexpr int MAXW = 16384, MAXH = 16384;
    for (;;) {
        std::vector<int> sky(W, 0);
        int Hmax = 0;
        bool ok = true;
        for (auto& p : patches) {
            int tw = p.tex_w, th = p.tex_h;
            int best_x = -1, best_y = MAXH;
            for (int x = 0; x + tw <= W; ++x) {
                int h = 0;
                for (int k = 0; k < tw; ++k) h = std::max(h, sky[x + k]);
                if (h < best_y) { best_y = h; best_x = x; }
            }
            if (best_x < 0 || best_y + th > MAXH) { ok = false; break; }
            p.atlas_x = best_x; p.atlas_y = best_y;
            for (int k = 0; k < tw; ++k) sky[best_x + k] = best_y + th;
            Hmax = std::max(Hmax, best_y + th);
        }
        if (ok) { H = Hmax; return; }
        if (W >= MAXW) { W = 1024; H = 0; return; }
        W *= 2;
    }
}

// ======================================================================
// main GPU pipeline
// ======================================================================

static GpuResult run_gpu_pipeline(const gfx::Model& model,
                                  const gfx::CoverageAtlasConfig& cfg) {
    printf("\n=== GPU MDC pipeline (example 34) ===\n");
    Pipe p;

    // ---- collect + weld submeshes on the host ---------------------------
    std::vector<glm::vec3> pos, nrm;
    std::vector<glm::vec2> uv;
    std::vector<uint32_t> idx;
    std::vector<uint32_t> triSub;
    glm::vec3 bbox_min(1e30f), bbox_max(-1e30f);

    {
        uint32_t vbase = 0;
        for (size_t m = 0; m < model.mesh_count(); ++m) {
            const auto& mesh = model.mesh(m);
            if (mesh.index_count() == 0) continue;
            std::vector<uint32_t> midx(mesh.indices().begin(), mesh.indices().end());
            std::vector<glm::vec3> mpos, mnrm;
            std::vector<glm::vec2> muv;
            float extent = 1.0f;
            for (const auto& v : mesh.vertices()) {
                for (int k = 0; k < 3; ++k) {
                    bbox_min[k] = std::min(bbox_min[k], v.position[k]);
                    bbox_max[k] = std::max(bbox_max[k], v.position[k]);
                }
            }
            extent = glm::length(bbox_max - bbox_min);
            if (extent < 1e-6f) extent = 1.0f;
            weld_positions(mesh.vertices(), mpos, mnrm, muv, midx,
                           extent * 1e-5f);
            p.T += uint32_t(midx.size() / 3);
            for (uint32_t& i : midx) i += vbase;
            pos.insert(pos.end(), mpos.begin(), mpos.end());
            nrm.insert(nrm.end(), mnrm.begin(), mnrm.end());
            uv.insert(uv.end(), muv.begin(), muv.end());
            idx.insert(idx.end(), midx.begin(), midx.end());
            for (size_t k = 0; k < midx.size() / 3; ++k) triSub.push_back(p.S);
            vbase = uint32_t(pos.size());
            p.S++;
        }
        if (p.T == 0) { fprintf(stderr, "model has no geometry\n"); std::exit(1); }
    }
    p.V = uint32_t(pos.size());

    // ---- upload geometry ------------------------------------------------
    p.b_pos = make_buf(pos.size() * 12);
    p.b_idx = make_buf(idx.size() * 4);
    p.b_uv  = make_buf(uv.size() * 8);
    upload(p.b_pos, pos); upload(p.b_idx, idx); upload(p.b_uv, uv);
    p.b_triSub = make_buf(p.T * 4);  upload(p.b_triSub, triSub);

    // ---- crease normals (GPU) -------------------------------------------
    p.b_faceN   = make_buf(p.T * 12);
    p.b_incPos  = make_buf(p.V * 4);
    p.b_incOff  = make_buf(p.V * 4);
    p.b_incFill = make_buf(p.V * 4);
    p.b_incFace = make_buf(p.T * 36);
    p.b_cornNrm = make_buf(p.T * 36);
    fill_u32(p.b_incPos, 0);

    double t_crease = run_ms([&] {
        p.b_faceN.bind_base(0); p.b_idx.bind_base(1); p.b_faceN.bind_base(2);
        p.k_face.prog.use();
        p.k_face.prog.uniform1ui(p.k_face.prog.uniform_location("uTri"), p.T);
        dispatch(p.k_face, p.T);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        p.b_idx.bind_base(0); p.b_incPos.bind_base(1);
        p.k_inccount.prog.use();
        p.k_inccount.prog.uniform1ui(p.k_inccount.prog.uniform_location("uTri"), p.T);
        dispatch(p.k_inccount, p.T);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        std::vector<uint32_t> incPos(p.V);
        readback(p.b_incPos, incPos);
        std::vector<uint32_t> incOff(p.V);
        uint32_t acc = 0;
        for (uint32_t i = 0; i < p.V; ++i) { incOff[i] = acc; acc += incPos[i]; }
        upload(p.b_incOff, incOff);

        fill_u32(p.b_incFill, 0);
        p.b_idx.bind_base(0); p.b_faceN.bind_base(1); p.b_incOff.bind_base(2);
        p.b_incFill.bind_base(3); p.b_incFace.bind_base(4);
        p.k_incfill.prog.use();
        p.k_incfill.prog.uniform1ui(p.k_incfill.prog.uniform_location("uTri"), p.T);
        dispatch(p.k_incfill, p.T);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        float crease_cos = std::cos(cfg.normal_crease_angle * float(M_PI) / 180.0f);
        p.b_idx.bind_base(0); p.b_faceN.bind_base(1); p.b_incOff.bind_base(2);
        p.b_incPos.bind_base(3); p.b_incFace.bind_base(4); p.b_cornNrm.bind_base(5);
        p.k_crease.prog.use();
        p.k_crease.prog.uniform1ui(p.k_crease.prog.uniform_location("uTri"), p.T);
        p.k_crease.prog.uniform1f(p.k_crease.prog.uniform_location("uCos"), crease_cos);
        dispatch(p.k_crease, p.T);
    });
    p.timings["crease normals"] = t_crease;
    printf("  crease normals: %8.2f ms\n", t_crease);

    // ---- per-triangle axis + submesh AABBs (GPU) -------------------------
    p.b_triAxis  = make_buf(p.T * 4);
    p.b_subAABB  = make_buf(p.S * 24);
    {
        std::vector<uint32_t> subaabb_init(p.S * 6, 0);
        for (uint32_t i = 0; i < p.S; ++i) {
            subaabb_init[i*6 + 0] = 0xFFFFFFFFu;
            subaabb_init[i*6 + 1] = 0xFFFFFFFFu;
            subaabb_init[i*6 + 2] = 0xFFFFFFFFu;
        }
        upload(p.b_subAABB, subaabb_init);
    }

    p.b_pos.bind_base(0); p.b_idx.bind_base(1); p.b_triSub.bind_base(2);
    p.b_faceN.bind_base(3); p.b_triAxis.bind_base(4); p.b_subAABB.bind_base(5);
    p.k_triprep.prog.use();
    p.k_triprep.prog.uniform1ui(p.k_triprep.prog.uniform_location("uTri"), p.T);
    double t_prep = run_ms([&] { dispatch(p.k_triprep, p.T); });
    p.timings["tri prep"] = t_prep;
    printf("  tri prep:       %8.2f ms\n", t_prep);

    // submesh AABB -> grid params on host
    std::vector<uint32_t> subraw(p.S * 6);
    readback(p.b_subAABB, subraw);
    std::vector<glm::vec4> grid(p.S * 6, glm::vec4(0, 0, 0, 1));
    glm::vec3 model_min(1e30f), model_max(-1e30f);
    for (uint32_t s = 0; s < p.S; ++s) {
        bool ok = true;
        glm::vec3 mn(1e30f), mx(-1e30f);
        for (int k = 0; k < 3; ++k) {
            float lo = s2f(subraw[s*6 + k]);
            float hi = s2f(subraw[s*6 + 3 + k]);
            if (subraw[s*6 + k] == 0xFFFFFFFFu || subraw[s*6 + 3 + k] == 0) { ok = false; break; }
            mn[k] = lo; mx[k] = hi;
        }
        if (!ok) continue;
        for (int axis = 0; axis < 6; ++axis) {
            auto b = axis_basis(axis);
            float uLo = proj_min(mn, mx, b.u), uHi = proj_max(mn, mx, b.u);
            float vLo = proj_min(mn, mx, b.v), vHi = proj_max(mn, mx, b.v);
            float pLo = proj_min(mn, mx, b.d);
            float cell = std::max(uHi - uLo, vHi - vLo) / float(GRID);
            if (cell < 1e-9f) cell = 1e-9f;
            grid[s*6 + axis] = glm::vec4(uLo, vLo, pLo, cell);
        }
        model_min = glm::min(model_min, mn);
        model_max = glm::max(model_max, mx);
    }
    p.b_grid = make_buf(grid.size() * 16);
    upload(p.b_grid, grid);

    // ---- clustering: cell union-find (GPU) ------------------------------
    uint32_t cellTotal = p.S * 6 * UCELL;
    p.b_cellMin  = make_buf(cellTotal * 4);
    p.b_cellMax  = make_buf(cellTotal * 4);
    p.b_cellOwn  = make_buf(cellTotal * 4);
    p.b_cellMerg = make_buf(cellTotal * 4);
    p.b_parent   = make_buf(p.T * 4);
    p.b_isol     = make_buf(p.T * 4);
    p.b_patchSize= make_buf(p.T * 4);
    p.b_viol     = make_buf(p.T * 4);
    p.b_best     = make_buf(p.T * 4);
    fill_u32(p.b_cellMin, 0xFFFFFFFFu);
    fill_u32(p.b_cellOwn, 0xFFFFFFFFu);
    fill_u32(p.b_cellMerg, 0xFFFFFFFFu);
    fill_u32(p.b_viol, 0);
    fill_u32(p.b_best, 0xFFFFFFFFu);
    {
        std::vector<uint32_t> init(p.T);
        for (uint32_t i = 0; i < p.T; ++i) init[i] = i;
        upload(p.b_parent, init);
    }
    fill_u32(p.b_patchSize, 0);
    fill_u32(p.b_isol, 0);

    auto bind_cells = [&](Kern& k, bool extra_isol = false) {
        p.b_pos.bind_base(0); p.b_idx.bind_base(1); p.b_triAxis.bind_base(2);
        p.b_triSub.bind_base(3); p.b_grid.bind_base(4);
        k.prog.use();
        k.prog.uniform1ui(k.prog.uniform_location("uTri"), p.T);
        k.prog.uniform1ui(k.prog.uniform_location("uMaxFoot"), MAX_FOOT);
        (void)extra_isol;
    };

    double t_cluster = run_ms([&] {
        bind_cells(p.k_cells);
        p.b_cellMin.bind_base(5); p.b_cellMax.bind_base(6); p.b_isol.bind_base(7);
        dispatch(p.k_cells, p.T);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        bind_cells(p.k_cellunion);
        p.b_cellMin.bind_base(5); p.b_cellMax.bind_base(6); p.b_cellOwn.bind_base(7);
        p.b_parent.bind_base(8); p.b_isol.bind_base(9);
        p.k_cellunion.prog.uniform1f(p.k_cellunion.prog.uniform_location("uEps"), cfg.epsilon);
        dispatch(p.k_cellunion, p.T);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        for (int it = 0; it < int(FIX_ITERS); ++it) {
            p.b_parent.bind_base(0);
            p.k_fix.prog.use();
            p.k_fix.prog.uniform1ui(p.k_fix.prog.uniform_location("uTri"), p.T);
            dispatch(p.k_fix, p.T);
            gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    });
    p.timings["clustering"] = t_cluster;
    printf("  clustering:     %8.2f ms\n", t_cluster);

    {
        std::vector<uint32_t> cmin(cellTotal), cmax(cellTotal);
        readback(p.b_cellMin, cmin);
        readback(p.b_cellMax, cmax);
        uint64_t occFree = 0, populated = 0;
        for (uint32_t i = 0; i < cellTotal; ++i) {
            if (cmin[i] == 0xFFFFFFFFu || cmax[i] == 0) continue;
            populated++;
            if (s2f(cmax[i]) - s2f(cmin[i]) <= cfg.epsilon) occFree++;
        }
        printf("  debug cells: populated=%llu occFree=%llu\n",
               (unsigned long long)populated, (unsigned long long)occFree);
        std::vector<uint32_t> par(p.T);
        readback(p.b_parent, par);
        std::vector<uint32_t> seen(p.T, 0);
        uint32_t roots = 0;
        for (uint32_t g = 0; g < p.T; ++g) if (seen[par[g]] == 0) { seen[par[g]] = 1; roots++; }
        printf("  debug roots after clustering: %u\n", roots);
    }

    // ---- small-patch merge (GPU) ----------------------------------------
    uint32_t merge_iters = 4;
    double t_merge = run_ms([&] {
        for (uint32_t it = 0; it < merge_iters; ++it) {
            fill_u32(p.b_patchSize, 0);
            p.b_parent.bind_base(0); p.b_patchSize.bind_base(1);
            p.k_sizes.prog.use();
            p.k_sizes.prog.uniform1ui(p.k_sizes.prog.uniform_location("uTri"), p.T);
            dispatch(p.k_sizes, p.T);
            gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

            fill_u32(p.b_cellMerg, 0xFFFFFFFFu);
            fill_u32(p.b_viol, 0);
            fill_u32(p.b_best, 0xFFFFFFFFu);

            bind_cells(p.k_mergea);
            p.b_cellMin.bind_base(5); p.b_cellMax.bind_base(6); p.b_cellMerg.bind_base(7);
            p.b_parent.bind_base(8); p.b_patchSize.bind_base(9); p.b_isol.bind_base(10);
            p.k_mergea.prog.uniform1f(p.k_mergea.prog.uniform_location("uEps"), cfg.epsilon);
            p.k_mergea.prog.uniform1ui(p.k_mergea.prog.uniform_location("uMinPatch"), cfg.min_patch_size);
            dispatch(p.k_mergea, p.T);
            gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

            bind_cells(p.k_mergeb);
            p.b_cellMin.bind_base(5); p.b_cellMax.bind_base(6); p.b_cellMerg.bind_base(7);
            p.b_parent.bind_base(8); p.b_patchSize.bind_base(9); p.b_viol.bind_base(10);
            p.b_best.bind_base(11); p.b_isol.bind_base(12);
            p.k_mergeb.prog.uniform1f(p.k_mergeb.prog.uniform_location("uEps"), cfg.epsilon);
            p.k_mergeb.prog.uniform1ui(p.k_mergeb.prog.uniform_location("uMinPatch"), cfg.min_patch_size);
            dispatch(p.k_mergeb, p.T);
            gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

            p.b_patchSize.bind_base(0); p.b_viol.bind_base(1); p.b_best.bind_base(2);
            p.b_parent.bind_base(3);
            p.k_mergec.prog.use();
            p.k_mergec.prog.uniform1ui(p.k_mergec.prog.uniform_location("uTri"), p.T);
            p.k_mergec.prog.uniform1ui(p.k_mergec.prog.uniform_location("uMinPatch"), cfg.min_patch_size);
            dispatch(p.k_mergec, p.T);
            gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

            for (int f = 0; f < int(FIX_ITERS); ++f) {
                p.b_parent.bind_base(0);
                p.k_fix.prog.use();
                p.k_fix.prog.uniform1ui(p.k_fix.prog.uniform_location("uTri"), p.T);
                dispatch(p.k_fix, p.T);
                gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }
        }
    });
    p.timings["merge"] = t_merge;
    printf("  merge:          %8.2f ms\n", t_merge);

    // ---- occlusion invariant spot-check ---------------------------------
    p.b_occ = make_buf(4);
    fill_u32(p.b_occ, 0);
    p.b_cellMin.bind_base(0); p.b_cellMax.bind_base(1); p.b_occ.bind_base(2);
    p.k_occ.prog.use();
    p.k_occ.prog.uniform1ui(p.k_occ.prog.uniform_location("uCells"), cellTotal);
    p.k_occ.prog.uniform1f(p.k_occ.prog.uniform_location("uEps"), cfg.epsilon);
    dispatch(p.k_occ, cellTotal);
    std::vector<uint32_t> occraw(1);
    readback(p.b_occ, occraw);
    float occ_max = occraw[0] == 0 ? 0.0f : s2f(occraw[0]);

    // ---- finalize patches (GPU) -----------------------------------------
    p.b_pAABBmin = make_buf(p.T * 12);
    p.b_pAABBmax = make_buf(p.T * 12);
    p.b_patchAxis = make_buf(p.T * 4);
    fill_u32(p.b_pAABBmin, 0xFFFFFFFFu);
    fill_u32(p.b_pAABBmax, 0);
    fill_u32(p.b_patchAxis, 0);

    double t_fin = run_ms([&] {
        p.b_parent.bind_base(0); p.b_pos.bind_base(1); p.b_idx.bind_base(2);
        p.b_triAxis.bind_base(3); p.b_patchAxis.bind_base(4);
        p.b_pAABBmin.bind_base(5); p.b_pAABBmax.bind_base(6);
        p.k_finalize.prog.use();
        p.k_finalize.prog.uniform1ui(p.k_finalize.prog.uniform_location("uTri"), p.T);
        dispatch(p.k_finalize, p.T);
    });
    p.timings["finalize"] = t_fin;
    printf("  finalize:       %8.2f ms\n", t_fin);

    // ---- host: labels + patches + sizing + pack -------------------------
    std::vector<uint32_t> parentArr(p.T), pminArr(p.T*3), pmaxArr(p.T*3), paxisArr(p.T);
    readback(p.b_parent, parentArr);
    readback(p.b_pAABBmin, pminArr);
    readback(p.b_pAABBmax, pmaxArr);
    readback(p.b_patchAxis, paxisArr);

    std::unordered_map<uint32_t, uint32_t> rootToLabel;
    std::vector<uint32_t> triLabel(p.T);
    std::vector<uint32_t> roots;
    for (uint32_t g = 0; g < p.T; ++g) {
        uint32_t r = parentArr[g];
        auto it = rootToLabel.find(r);
        uint32_t lab;
        if (it == rootToLabel.end()) {
            lab = uint32_t(roots.size());
            roots.push_back(r);
            rootToLabel[r] = lab;
        } else lab = it->second;
        triLabel[g] = lab;
    }
    p.P = uint32_t(roots.size());

    std::vector<PatchCpu> patches(p.P);
    for (uint32_t i = 0; i < p.P; ++i) {
        uint32_t r = roots[i];
        glm::vec3 mn(s2f(pminArr[r*3]), s2f(pminArr[r*3+1]), s2f(pminArr[r*3+2]));
        glm::vec3 mx(s2f(pmaxArr[r*3]), s2f(pmaxArr[r*3+1]), s2f(pmaxArr[r*3+2]));
        int axis = int(paxisArr[r]);
        auto b = axis_basis(axis);
        float uLo = proj_min(mn, mx, b.u), uHi = proj_max(mn, mx, b.u);
        float vLo = proj_min(mn, mx, b.v), vHi = proj_max(mn, mx, b.v);
        patches[i].aabb_min = mn; patches[i].aabb_max = mx;
        patches[i].axis = axis;
        patches[i].proj_u_lo = uLo; patches[i].proj_v_lo = vLo;
        patches[i].proj_u_size = std::max(uHi - uLo, 1e-9f);
        patches[i].proj_v_size = std::max(vHi - vLo, 1e-9f);
    }

    float density = cfg.texel_density;
    if (density <= 0) {
        float span = glm::length(model_max - model_min);
        density = float(cfg.auto_target) / std::max(span, 1e-6f);
    }
    size_textures(patches, density, cfg.min_tex, cfg.max_tex);
    fit_to_budget(patches, density, cfg.budget_texels, cfg.min_tex, cfg.max_tex);

    int atlas_w, atlas_h;
    pack_atlas(patches, atlas_w, atlas_h);
    p.atlas_w = uint32_t(atlas_w); p.atlas_h = uint32_t(atlas_h);
    printf("  atlas:          %d x %d texels\n", atlas_w, atlas_h);

    // ---- upload per-patch params + labels -------------------------------
    std::vector<float> pabb(p.P * 6);
    std::vector<uint32_t> rect(p.P * 4), pN(p.P);
    std::vector<uint32_t> paxisC(p.P);
    for (uint32_t i = 0; i < p.P; ++i) {
        pabb[i*6+0] = patches[i].aabb_min.x; pabb[i*6+1] = patches[i].aabb_min.y; pabb[i*6+2] = patches[i].aabb_min.z;
        pabb[i*6+3] = patches[i].aabb_max.x; pabb[i*6+4] = patches[i].aabb_max.y; pabb[i*6+5] = patches[i].aabb_max.z;
        rect[i*4+0] = uint32_t(patches[i].atlas_x);
        rect[i*4+1] = uint32_t(patches[i].atlas_y);
        rect[i*4+2] = uint32_t(patches[i].tex_w);
        rect[i*4+3] = uint32_t(patches[i].tex_h);
        pN[i] = uint32_t(next_pow2_int(std::max(patches[i].tex_w, patches[i].tex_h)));
        paxisC[i] = uint32_t(patches[i].axis);
    }
    p.b_patchAABB = make_buf(pabb.size() * 4);
    p.b_patchRect = make_buf(rect.size() * 4);
    p.b_patchN    = make_buf(pN.size() * 4);
    p.b_patchAxis = make_buf(paxisC.size() * 4);
    upload(p.b_patchAABB, pabb); upload(p.b_patchRect, rect); upload(p.b_patchN, pN);
    upload(p.b_patchAxis, paxisC);
    p.b_triLabel = make_buf(p.T * 4);
    upload(p.b_triLabel, triLabel);

    // ---- triangle reorder (patch-major) ---------------------------------
    p.b_patchTriCnt  = make_buf(p.P * 4);
    p.b_patchTriOff  = make_buf(p.P * 4);
    p.b_triSorted    = make_buf(p.T * 4);
    fill_u32(p.b_patchTriCnt, 0);
    std::vector<uint32_t> triOff(p.P, 0);
    {
        p.b_triLabel.bind_base(0); p.b_patchTriCnt.bind_base(1);
        p.k_reorderc.prog.use();
        p.k_reorderc.prog.uniform1ui(p.k_reorderc.prog.uniform_location("uTri"), p.T);
        dispatch(p.k_reorderc, p.T);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        std::vector<uint32_t> cnt(p.P);
        readback(p.b_patchTriCnt, cnt);
        uint32_t acc = 0;
        for (uint32_t i = 0; i < p.P; ++i) {
            triOff[i] = acc;
            acc += cnt[i];
            patches[i].tri_count = cnt[i];
            patches[i].tri_offset = triOff[i];
        }
        upload(p.b_patchTriOff, triOff);

        p.b_triLabel.bind_base(0); p.b_patchTriOff.bind_base(1); p.b_triSorted.bind_base(2);
        p.k_reorderf.prog.use();
        p.k_reorderf.prog.uniform1ui(p.k_reorderf.prog.uniform_location("uTri"), p.T);
        dispatch(p.k_reorderf, p.T);
    }

    // ---- atlas buffers + rasterise --------------------------------------
    uint32_t atlW = uint32_t(atlas_w), atlH = uint32_t(atlas_h);
    uint32_t atlN = atlW * atlH;
    p.b_atlasDmin   = make_buf(atlN * 4);
    p.b_atlasDmax   = make_buf(atlN * 4);
    p.b_atlasThick  = make_buf(atlN * 4);
    p.b_atlasUv     = make_buf(atlN * 8);
    p.b_atlasNrm    = make_buf(atlN * 8);
    p.b_atlasRect   = make_buf(atlN * 4);
    p.b_atlasState  = make_buf(atlN * 4);
    p.b_changed     = make_buf(4);
    fill_u32(p.b_atlasDmin, 0xFFFFFFFFu);
    fill_u32(p.b_atlasRect, 0xFFFFFFFFu);
    fill_u32(p.b_atlasState, 0);
    // dmax/thick/uv/nrm default 0
    fill_u32(p.b_atlasDmax, 0); fill_u32(p.b_atlasThick, 0);

    double t_rast = run_ms([&] {
        p.b_pos.bind_base(0); p.b_idx.bind_base(1); p.b_uv.bind_base(2);
        p.b_cornNrm.bind_base(3); p.b_triLabel.bind_base(4); p.b_patchAxis.bind_base(5);
        p.b_patchAABB.bind_base(6); p.b_patchRect.bind_base(7); p.b_patchN.bind_base(8);
        p.b_atlasDmin.bind_base(9); p.b_atlasDmax.bind_base(10); p.b_atlasThick.bind_base(11);
        p.b_atlasUv.bind_base(12); p.b_atlasNrm.bind_base(13);
        p.k_raster.prog.use();
        p.k_raster.prog.uniform1ui(p.k_raster.prog.uniform_location("uTri"), p.T);
        p.k_raster.prog.uniform1ui(p.k_raster.prog.uniform_location("uW"), atlW);
        dispatch(p.k_raster, p.T);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    });
    p.timings["rasterize"] = t_rast;
    printf("  rasterize:      %8.2f ms\n", t_rast);

    // ---- hole filling ----------------------------------------------------
    double t_hole = run_ms([&] {
        p.b_patchRect.bind_base(0); p.b_atlasRect.bind_base(1);
        p.k_hf_setup.prog.use();
        p.k_hf_setup.prog.uniform1ui(p.k_hf_setup.prog.uniform_location("uPatches"), p.P);
        p.k_hf_setup.prog.uniform1ui(p.k_hf_setup.prog.uniform_location("uW"), atlW);
        dispatch(p.k_hf_setup, p.P);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        p.b_patchRect.bind_base(0); p.b_atlasDmin.bind_base(1); p.b_atlasState.bind_base(2);
        p.k_hf_border.prog.use();
        p.k_hf_border.prog.uniform1ui(p.k_hf_border.prog.uniform_location("uPatches"), p.P);
        p.k_hf_border.prog.uniform1ui(p.k_hf_border.prog.uniform_location("uW"), atlW);
        dispatch(p.k_hf_border, p.P);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        auto flood = [&] {
            for (int iter = 0; iter < 128; ++iter) {
                fill_u32(p.b_changed, 0);
                p.b_patchRect.bind_base(0); p.b_atlasRect.bind_base(1);
                p.b_atlasDmin.bind_base(2); p.b_atlasState.bind_base(3); p.b_changed.bind_base(4);
                p.k_hf_flood.prog.use();
                p.k_hf_flood.prog.uniform1ui(p.k_hf_flood.prog.uniform_location("uW"), atlW);
                p.k_hf_flood.prog.uniform1ui(p.k_hf_flood.prog.uniform_location("uH"), atlH);
                dispatch(p.k_hf_flood, atlN);
                gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
                uint32_t ch = 1;
                readback(p.b_changed, &ch, 4);
                if (!ch) break;
            }
        };
        auto fill = [&] {
            for (int iter = 0; iter < 128; ++iter) {
                fill_u32(p.b_changed, 0);
                p.b_patchRect.bind_base(0); p.b_atlasRect.bind_base(1);
                p.b_atlasDmin.bind_base(2); p.b_atlasDmax.bind_base(3); p.b_atlasThick.bind_base(4);
                p.b_atlasUv.bind_base(5); p.b_atlasNrm.bind_base(6);
                p.b_atlasState.bind_base(7); p.b_changed.bind_base(8);
                p.k_hf_fill.prog.use();
                p.k_hf_fill.prog.uniform1ui(p.k_hf_fill.prog.uniform_location("uW"), atlW);
                p.k_hf_fill.prog.uniform1ui(p.k_hf_fill.prog.uniform_location("uH"), atlH);
                dispatch(p.k_hf_fill, atlN);
                gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
                uint32_t ch = 1;
                readback(p.b_changed, &ch, 4);
                if (!ch) break;
            }
        };
        flood();
        fill();
    });
    p.timings["hole fill"] = t_hole;
    printf("  hole fill:      %8.2f ms\n", t_hole);

    // ---- mip chains ------------------------------------------------------
    p.b_mipRange = make_buf(8 * 4);
    std::vector<uint32_t> mipRangeInit = {0xFFFFFFFFu, 0, 0, 0xFFFFFFFFu, 0, 0xFFFFFFFFu, 0, 0};
    upload(p.b_mipRange, mipRangeInit);
    {
        p.b_atlasDmin.bind_base(0); p.b_atlasDmax.bind_base(1); p.b_atlasThick.bind_base(2);
        p.b_atlasUv.bind_base(3); p.b_mipRange.bind_base(4);
        p.k_range.prog.use();
        p.k_range.prog.uniform1ui(p.k_range.prog.uniform_location("uW"), atlW);
        p.k_range.prog.uniform1ui(p.k_range.prog.uniform_location("uH"), atlH);
        dispatch(p.k_range, atlN);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    std::vector<uint32_t> rg(8);
    readback(p.b_mipRange, rg);
    float gDmin = s2f(rg[0]), gDmax = s2f(rg[1]);
    float gThick = s2f(rg[2]), gUmin = s2f(rg[3]), gUmax = s2f(rg[4]);
    float gVmin = s2f(rg[5]), gVmax = s2f(rg[6]);

    p.b_mipN0   = make_buf(MIP_CAP * 4);
    p.b_mipN1   = make_buf(MIP_CAP * 4);
    p.b_mipN0b  = make_buf(MIP_CAP * 4);
    p.b_mipN1b  = make_buf(MIP_CAP * 4);
    p.b_mipVal  = make_buf(MIP_CAP * 8);
    p.b_mipFlag = make_buf(MIP_CAP * 4);
    p.b_mipNext = make_buf(4);
    fill_u32(p.b_mipNext, 0);

    // level 0 = one node per patch
    std::vector<uint32_t> n0(p.P), n1(p.P);
    for (uint32_t i = 0; i < p.P; ++i) { n0[i] = i; n1[i] = 0; }
    uint32_t nodeCount = p.P;
    uint32_t valBase = 0;
    std::vector<uint32_t> mipLevelCounts;

    double t_mip = run_ms([&] {
        gl::Buffer* readA0 = &p.b_mipN0; gl::Buffer* readA1 = &p.b_mipN1;
        gl::Buffer* writeB0 = &p.b_mipN0b; gl::Buffer* writeB1 = &p.b_mipN1b;
        upload(*readA0, n0); upload(*readA1, n1);
        for (uint32_t level = 0; level < MAX_LEVELS && nodeCount > 0; ++level) {
            fill_u32(p.b_mipNext, 0);
            readA0->bind_base(0); readA1->bind_base(1);
            p.b_patchRect.bind_base(2); p.b_patchN.bind_base(3);
            p.b_atlasDmin.bind_base(4); p.b_atlasDmax.bind_base(5); p.b_atlasThick.bind_base(6);
            p.b_atlasUv.bind_base(7); p.b_atlasNrm.bind_base(8);
            p.b_mipVal.bind_base(9); p.b_mipFlag.bind_base(10); p.b_mipNext.bind_base(11);
            writeB0->bind_base(12); writeB1->bind_base(13);
            p.k_mipagg.prog.use();
            p.k_mipagg.prog.uniform1ui(p.k_mipagg.prog.uniform_location("uLevel"), level);
            p.k_mipagg.prog.uniform1ui(p.k_mipagg.prog.uniform_location("uNodeCount"), nodeCount);
            p.k_mipagg.prog.uniform1ui(p.k_mipagg.prog.uniform_location("uValBase"), valBase);
            p.k_mipagg.prog.uniform1ui(p.k_mipagg.prog.uniform_location("uW"), atlW);
            p.k_mipagg.prog.uniform1ui(p.k_mipagg.prog.uniform_location("uLeaf"), uint32_t(cfg.mip_leaf_tile));
            p.k_mipagg.prog.uniform1f(p.k_mipagg.prog.uniform_location("uDepthTol"),
                                      (gDmax - gDmin) * cfg.mip_tol_frac);
            p.k_mipagg.prog.uniform1f(p.k_mipagg.prog.uniform_location("uThickTol"),
                                      gThick * cfg.mip_tol_frac);
            p.k_mipagg.prog.uniform1f(p.k_mipagg.prog.uniform_location("uUvTol"),
                                      std::max(std::max(gUmax - gUmin, gVmax - gVmin), 0.0f) * cfg.mip_tol_frac);
            p.k_mipagg.prog.uniform1f(p.k_mipagg.prog.uniform_location("uDmin"), gDmin);
            p.k_mipagg.prog.uniform1f(p.k_mipagg.prog.uniform_location("uDmax"), gDmax);
            p.k_mipagg.prog.uniform1f(p.k_mipagg.prog.uniform_location("uThickMax"), gThick);
            p.k_mipagg.prog.uniform1f(p.k_mipagg.prog.uniform_location("uUmin"), gUmin);
            p.k_mipagg.prog.uniform1f(p.k_mipagg.prog.uniform_location("uUmax"), gUmax);
            p.k_mipagg.prog.uniform1f(p.k_mipagg.prog.uniform_location("uVmin"), gVmin);
            p.k_mipagg.prog.uniform1f(p.k_mipagg.prog.uniform_location("uVmax"), gVmax);
            dispatch(p.k_mipagg, nodeCount);
            gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

            uint32_t next = 0;
            readback(p.b_mipNext, &next, 4);
            mipLevelCounts.push_back(nodeCount);
            valBase += nodeCount;
            nodeCount = next * 4;
            std::swap(readA0, writeB0);
            std::swap(readA1, writeB1);
        }
    });
    p.timings["mip chains"] = t_mip;
    printf("  mip chains:     %8.2f ms\n", t_mip);

    // ---- BVH -------------------------------------------------------------
    p.b_morton   = make_buf(p.P * 4);
    p.b_sortedP  = make_buf(p.P * 4);
    p.b_mbb      = make_buf(6 * 4);
    {
        std::vector<float> mbb = {model_min.x, model_min.y, model_min.z,
                                  model_max.x, model_max.y, model_max.z};
        upload(p.b_mbb, mbb);
        p.b_patchAABB.bind_base(0); p.b_mbb.bind_base(1); p.b_morton.bind_base(2);
        p.k_morton.prog.use();
        p.k_morton.prog.uniform1ui(p.k_morton.prog.uniform_location("uPatches"), p.P);
        dispatch(p.k_morton, p.P);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // radix sort (4 passes, ping-pong)
        p.b_rk0 = make_buf(p.P * 4);
        p.b_rk1 = make_buf(p.P * 4);
        p.b_ra0 = make_buf(p.P * 4);
        p.b_ra1 = make_buf(p.P * 4);
        p.b_rcounts = make_buf(256 * 4);
        std::vector<uint32_t> ident(p.P);
        for (uint32_t i = 0; i < p.P; ++i) ident[i] = i;
        upload(p.b_ra0, ident);
        std::vector<uint32_t> morton(p.P);
        readback(p.b_morton, morton);
        upload(p.b_rk0, morton);

        gl::Buffer* kin = &p.b_rk0; gl::Buffer* kout = &p.b_rk1;
        gl::Buffer* ain = &p.b_ra0; gl::Buffer* aout = &p.b_ra1;
        for (uint32_t pass = 0; pass < 4; ++pass) {
            fill_u32(p.b_rcounts, 0);
            kin->bind_base(0); p.b_rcounts.bind_base(1);
            p.k_radix_count.prog.use();
            p.k_radix_count.prog.uniform1ui(p.k_radix_count.prog.uniform_location("uN"), p.P);
            p.k_radix_count.prog.uniform1ui(p.k_radix_count.prog.uniform_location("uPass"), pass);
            dispatch(p.k_radix_count, p.P);
            gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

            p.b_rcounts.bind_base(0);
            p.k_radix_scan.prog.use();
            dispatch(p.k_radix_scan, 1);
            gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

            kin->bind_base(0); ain->bind_base(1); kout->bind_base(2); aout->bind_base(3);
            p.b_rcounts.bind_base(4);
            p.k_radix_scatter.prog.use();
            p.k_radix_scatter.prog.uniform1ui(p.k_radix_scatter.prog.uniform_location("uN"), p.P);
            p.k_radix_scatter.prog.uniform1ui(p.k_radix_scatter.prog.uniform_location("uPass"), pass);
            dispatch(p.k_radix_scatter, p.P);
            gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
            std::swap(kin, kout);
            std::swap(ain, aout);
        }
        // sorted patches are in ain (or aout) — copy to b_sortedP
        std::vector<uint32_t> sortedP(p.P);
        readback(*ain, sortedP);
        upload(p.b_sortedP, sortedP);
    }
    double t_bvh = 0;
    {
        auto t0 = std::chrono::steady_clock::now();
        uint32_t P = p.P;
        p.b_bvRange   = make_buf(P * 2 * 8);
        p.b_bvChild   = make_buf(P * 2 * 8);
        p.b_bvNodeCnt = make_buf(4);
        p.b_bvLevelA  = make_buf(P * 2 * 4);
        p.b_bvLevelB  = make_buf(P * 2 * 4);
        p.b_bvNextCnt = make_buf(4);
        p.b_bvAABB    = make_buf(P * 2 * 24);
        p.b_bvSize    = make_buf(P * 2 * 4);
        p.b_bvOutA    = make_buf(P * 2 * 24);
        p.b_bvOutB    = make_buf(P * 2 * 8);
        fill_u32(p.b_bvNodeCnt, 1);
        fill_u32(p.b_bvNextCnt, 0);
        std::vector<int> rinit(P * 2, 0);
        rinit[0] = 0; rinit[1] = int(P) - 1;
        upload(p.b_bvRange, rinit);
        std::vector<uint32_t> lvl0 = {0};
        upload(p.b_bvLevelA, lvl0);

        std::vector<std::vector<uint32_t>> levels;
        levels.push_back(lvl0);
        gl::Buffer* la = &p.b_bvLevelA; gl::Buffer* lb = &p.b_bvLevelB;
        uint32_t lc = 1;
        for (;;) {
            fill_u32(p.b_bvNextCnt, 0);
            p.b_bvRange.bind_base(0); p.b_bvChild.bind_base(1); p.b_bvNodeCnt.bind_base(2);
            la->bind_base(3); lb->bind_base(4); p.b_bvNextCnt.bind_base(5);
            p.k_bvsplit.prog.use();
            p.k_bvsplit.prog.uniform1ui(p.k_bvsplit.prog.uniform_location("uLevelCount"), lc);
            dispatch(p.k_bvsplit, lc);
            gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
            uint32_t nc = 0;
            readback(p.b_bvNextCnt, &nc, 4);
            if (nc == 0) break;
            std::vector<uint32_t> lv(nc);
            readback(*lb, lv.data(), nc * 4);
            levels.push_back(std::move(lv));
            lc = nc;
            std::swap(la, lb);
        }

        // bottom-up AABB + subtree sizes
        for (size_t li = levels.size(); li-- > 0;) {
            std::vector<uint32_t> lv = levels[li];
            upload(p.b_bvLevelA, lv);
            p.b_bvLevelA.bind_base(0); p.b_bvRange.bind_base(1); p.b_bvChild.bind_base(2);
            p.b_patchAABB.bind_base(3); p.b_sortedP.bind_base(4);
            p.b_bvAABB.bind_base(5); p.b_bvSize.bind_base(6);
            p.k_bvaabb.prog.use();
            p.k_bvaabb.prog.uniform1ui(p.k_bvaabb.prog.uniform_location("uLevelCount"), uint32_t(lv.size()));
            dispatch(p.k_bvaabb, uint32_t(lv.size()));
            gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        p.b_bvRange.bind_base(0); p.b_bvChild.bind_base(1); p.b_sortedP.bind_base(2);
        p.b_bvAABB.bind_base(3); p.b_bvSize.bind_base(4);
        p.b_bvOutA.bind_base(5); p.b_bvOutB.bind_base(6);
        p.k_bvrenumber.prog.use();
        p.k_bvrenumber.prog.uniform1ui(p.k_bvrenumber.prog.uniform_location("uPatches"), P);
        dispatch(p.k_bvrenumber, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
        auto t1 = std::chrono::steady_clock::now();
        t_bvh = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    p.timings["bvh"] = t_bvh;
    printf("  bvh:            %8.2f ms\n", t_bvh);

    // ---- assemble result --------------------------------------------------
    GpuResult res;
    res.positions = pos;
    res.patch_count = p.P;
    res.tri_count = p.T;
    res.atlas_w = atlas_w; res.atlas_h = atlas_h;
    res.occ_max_over = occ_max;
    res.mip_level_counts = mipLevelCounts;
    res.bvh_nodes = p.P * 2 - 1;
    res.per_axis.assign(6, 0);
    for (auto& pt : patches) res.per_axis[pt.axis]++;
    res.patch_tri_counts.resize(p.P);
    res.patch_tri_offsets.resize(p.P);
    for (uint32_t i = 0; i < p.P; ++i) {
        res.patch_tri_counts[i] = patches[i].tri_count;
        res.patch_tri_offsets[i] = patches[i].tri_offset;
        res.patches_center.push_back((patches[i].aabb_min + patches[i].aabb_max) * 0.5f);
    }

    // coverage
    {
        std::vector<uint32_t> dmin(atlN);
        readback(p.b_atlasDmin, dmin);
        for (uint32_t i = 0; i < atlN; ++i) if (dmin[i] != 0xFFFFFFFFu) ++res.covered_texels;
        res.atlas_texels = atlN;
    }

    // render index buffer (patch-major)
    {
        std::vector<uint32_t> sorted(p.T);
        readback(p.b_triSorted, sorted);
        res.render_indices.reserve(p.T * 3);
        for (uint32_t i = 0; i < p.P; ++i) {
            for (uint32_t k = 0; k < patches[i].tri_count; ++k) {
                uint32_t tri = sorted[patches[i].tri_offset + k];
                res.render_indices.push_back(idx[3*tri]);
                res.render_indices.push_back(idx[3*tri+1]);
                res.render_indices.push_back(idx[3*tri+2]);
            }
        }
    }

    // smooth vertex normals from crease corner normals
    {
        std::vector<float> cn(p.T * 9);
        readback(p.b_cornNrm, cn);
        std::vector<glm::vec3> acc(p.V, glm::vec3(0));
        std::vector<uint32_t> cnt(p.V, 0);
        for (uint32_t g = 0; g < p.T; ++g) {
            for (int e = 0; e < 3; ++e) {
                uint32_t v = idx[3*g + e];
                uint32_t ci = 3*g + e;
                acc[v] += glm::vec3(cn[3*ci], cn[3*ci+1], cn[3*ci+2]);
                cnt[v]++;
            }
        }
        res.normals.resize(p.V);
        for (uint32_t v = 0; v < p.V; ++v) {
            res.normals[v] = cnt[v] ? glm::normalize(acc[v]) : glm::vec3(0, 0, 1);
        }
    }
    res.timings = p.timings;
    return res;
}

// ======================================================================
// report
// ======================================================================

static void print_report(const GpuResult& gpu, const gfx::CoverageAtlas& cpu,
                         double cpu_ms) {
    printf("\n=== GPU vs CPU pipeline diff ===\n");
    double gpu_total = 0;
    for (auto& [name, ms] : gpu.timings) gpu_total += ms;

    printf("Timing (GPU stages): %.2f ms total   |   CPU reference: %.2f ms\n",
           gpu_total, cpu_ms);

    const auto& cpu_patches = cpu.patches();
    uint64_t cpu_min = 0, cpu_max = 0, cpu_avg = 0;
    if (!cpu_patches.empty()) {
        cpu_min = cpu_patches[0].tris.size();
        cpu_max = cpu_patches[0].tris.size();
        for (auto& p : cpu_patches) {
            cpu_min = std::min<uint64_t>(cpu_min, p.tris.size());
            cpu_max = std::max<uint64_t>(cpu_max, p.tris.size());
            cpu_avg += p.tris.size();
        }
        cpu_avg /= cpu_patches.size();
    }
    uint64_t gpu_min = 0, gpu_max = 0, gpu_avg = 0;
    for (auto& c : gpu.patch_tri_counts) {
        gpu_min = gpu_min ? std::min<uint64_t>(gpu_min, c) : c;
        gpu_max = std::max<uint64_t>(gpu_max, c);
        gpu_avg += c;
    }
    if (gpu.patch_count) gpu_avg /= gpu.patch_count;

    printf("Patches:        GPU %u   CPU %zu\n", gpu.patch_count, cpu_patches.size());
    printf("Triangles:      GPU %u assigned   CPU %zu\n", gpu.tri_count, cpu.triangles().size());
    printf("Tris per patch: GPU min=%llu avg=%llu max=%llu   CPU min=%llu avg=%llu max=%llu\n",
           (unsigned long long)gpu_min, (unsigned long long)gpu_avg, (unsigned long long)gpu_max,
           (unsigned long long)cpu_min, (unsigned long long)cpu_avg, (unsigned long long)cpu_max);
    static const char* AXIS[] = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};
    printf("By axis (GPU): ");
    for (int i = 0; i < 6; ++i) printf(" %s=%u", AXIS[i], gpu.per_axis[i]);
    printf("\n");

    uint64_t cpu_covered = 0;
    for (auto b : cpu.coverage()) if (b) ++cpu_covered;
    printf("Atlas:          GPU %dx%d covered %.2f%%   CPU %dx%d covered %.2f%%\n",
           gpu.atlas_w, gpu.atlas_h,
           gpu.atlas_texels ? 100.0 * double(gpu.covered_texels) / double(gpu.atlas_texels) : 0.0,
           cpu.atlas_width(), cpu.atlas_height(),
           cpu.coverage().size() ? 100.0 * double(cpu_covered) / double(cpu.coverage().size()) : 0.0);

    size_t max_levels = std::max(gpu.mip_level_counts.size(), cpu.depth_chain().levels.size());
    printf("Mip nodes/level (GPU | CPU):\n");
    for (size_t L = 0; L < max_levels; ++L) {
        uint32_t g = L < gpu.mip_level_counts.size() ? gpu.mip_level_counts[L] : 0;
        uint32_t c = 0;
        if (L < cpu.depth_chain().levels.size())
            c = cpu.depth_chain().levels[L].w * cpu.depth_chain().levels[L].h;
        printf("  level %2zu: %8u | %8u\n", L, g, c);
    }

    printf("Occlusion-free (GPU): max cell depth range over epsilon = %g (%s)\n",
           gpu.occ_max_over, gpu.occ_max_over <= 0.0f ? "PASS" : "WARN");
    printf("BVH nodes:      GPU %u   CPU %zu\n", gpu.bvh_nodes, cpu.bvh_nodes().size());
}

// ======================================================================
// render (multi-draw indirect, ex31-style)
// ======================================================================

static const char* vert_src = R"(
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

static const char* frag_src = R"(
#version 460 core
in vec3 v_normal;
flat in int v_patch_id;
layout(std430, binding = 0) readonly buffer ColorBuf {
    vec4 patch_colors[];
};
out vec4 frag_color;
void main() {
    vec3 n = normalize(v_normal);
    vec3 light_dir = normalize(vec3(1.0, 2.0, 1.5));
    float diff = abs(dot(n, light_dir));
    float light = 0.55 + 0.45 * diff;
    frag_color = vec4(patch_colors[v_patch_id].rgb * light, 1.0);
}
)";

struct RenderVertex { float position[3]; float normal[3]; };

static void run_render(gfx::Window& window, const GpuResult& res) {
    std::vector<RenderVertex> render_verts(res.positions.size());
    for (size_t i = 0; i < res.positions.size(); ++i) {
        render_verts[i] = {
            {res.positions[i].x, res.positions[i].y, res.positions[i].z},
            {res.normals[i].x, res.normals[i].y, res.normals[i].z}
        };
    }

    std::vector<gl::DrawElementsIndirectCommand> draw_cmds;
    draw_cmds.reserve(res.patch_count);
    for (uint32_t i = 0; i < res.patch_count; ++i) {
        uint32_t count = res.patch_tri_counts[i] * 3;
        draw_cmds.push_back({count, 1, res.patch_tri_offsets[i] * 3, 0, 0});
    }

    struct alignas(16) PatchColor { float r, g, b, a; };
    std::vector<PatchColor> patch_colors(res.patch_count);
    {
        std::mt19937 rng(1337);
        std::uniform_real_distribution<float> s_dist(0.45f, 1.0f);
        std::uniform_real_distribution<float> v_dist(0.7f, 1.0f);
        auto rand_hue = [&rng]() {
            return std::fmod(std::fmod(float(rng()) * 2.3283064e-10f, 1.0f) + 1.0f, 1.0f);
        };
        for (size_t i = 0; i < patch_colors.size(); ++i) {
            float h = rand_hue(), s = s_dist(rng), v = v_dist(rng);
            float r, g, b;
            float c = v * s;
            float hp = h * 6.0f;
            float x = c * (1.0f - std::fabs(std::fmod(hp, 2.0f) - 1.0f));
            switch (int(hp)) {
                case 0: r = c; g = x; b = 0; break;
                case 1: r = x; g = c; b = 0; break;
                case 2: r = 0; g = c; b = x; break;
                case 3: r = 0; g = x; b = c; break;
                case 4: r = x; g = 0; b = c; break;
                default: r = c; g = 0; b = x; break;
            }
            float m = v - c;
            patch_colors[i] = {r + m, g + m, b + m, 1.0f};
        }
    }

    GLuint vao, vbo, ebo, indirect_buf, color_ssbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glGenBuffers(1, &indirect_buf);
    glGenBuffers(1, &color_ssbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, render_verts.size() * sizeof(RenderVertex),
                 render_verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, res.render_indices.size() * sizeof(GLuint),
                 res.render_indices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex),
                          (void*)offsetof(RenderVertex, normal));
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_buf);
    glBufferData(GL_DRAW_INDIRECT_BUFFER,
                 draw_cmds.size() * sizeof(gl::DrawElementsIndirectCommand),
                 draw_cmds.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, color_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, patch_colors.size() * sizeof(PatchColor),
                 patch_colors.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, color_ssbo);
    glBindVertexArray(0);

    gl::Shader vs(gl::ShaderType::vertex, vert_src);
    if (!vs.compiled()) { fprintf(stderr, "VS failed:\n%s\n", vs.info_log().c_str()); return; }
    gl::Shader fs(gl::ShaderType::fragment, frag_src);
    if (!fs.compiled()) { fprintf(stderr, "FS failed:\n%s\n", fs.info_log().c_str()); return; }
    gl::Program prog;
    prog.attach(vs); prog.attach(fs);
    if (!prog.link()) { fprintf(stderr, "Link failed:\n%s\n", prog.info_log().c_str()); return; }

    glm::vec3 scene_lo(1e30f), scene_hi(-1e30f);
    for (auto& p : res.positions) {
        scene_lo = glm::min(scene_lo, p);
        scene_hi = glm::max(scene_hi, p);
    }
    glm::vec3 scene_center = (scene_lo + scene_hi) * 0.5f;
    float scene_radius = glm::length(scene_hi - scene_lo) * 0.5f;

    gfx::Camera cam;
    cam.perspective(45.0f, float(window.width()) / float(window.height()),
                    scene_radius * 0.001f, scene_radius * 10.0f);
    cam.look_at(glm::vec3(scene_center.x + scene_radius * 2.0f,
                          scene_center.y + scene_radius * 1.0f,
                          scene_center.z + scene_radius * 2.0f),
                scene_center);

    double prev_x, prev_y;
    window.cursor_position(prev_x, prev_y);

    gfx::Renderer renderer;
    renderer.set_clear_color(0.05f, 0.05f, 0.1f, 1.0f);

    printf("Rendering %u patches, %zu draw commands\n", res.patch_count, draw_cmds.size());
    while (!window.should_close()) {
        window.poll_events();
        if (window.mouse_down(gfx::MouseButton::left)) {
            double cx, cy;
            window.cursor_position(cx, cy);
            float dx = float(cx - prev_x) * 0.005f;
            float dy = float(prev_y - cy) * 0.005f;
            cam.orbit(dx, dy);
            prev_x = cx; prev_y = cy;
        } else {
            window.cursor_position(prev_x, prev_y);
        }
        double scroll = window.scroll_delta();
        if (scroll != 0.0) cam.zoom(float(scroll) * scene_radius * 0.01f);
        cam.set_aspect(float(window.width()) / float(window.height()));

        gl::viewport(0, 0, window.width(), window.height());
        renderer.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        prog.use();
        glm::mat4 vp = cam.view_projection();
        glUniformMatrix4fv(prog.uniform_location("u_vp"), 1, GL_FALSE, &vp[0][0]);
        glBindVertexArray(vao);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, color_ssbo);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_buf);
        gl::multi_draw_elements_indirect(GL_TRIANGLES, GL_UNSIGNED_INT,
                                         nullptr, GLsizei(draw_cmds.size()), 0);
        window.swap_buffers();
    }
}

// ======================================================================
// main
// ======================================================================

int main(int argc, char** argv) {
    const char* model_path = "Stanford_Dragon.glb";
    gfx::CoverageAtlasConfig cfg;
    cfg.rotate_model_x = false;

    if (argc > 1) model_path = argv[1];
    if (argc > 2) cfg.texel_density = std::atof(argv[2]);
    if (argc > 3) cfg.budget_texels = std::atof(argv[3]) * 1e6f;
    if (argc > 4) cfg.min_patch_size = std::atoi(argv[4]);
    if (argc > 5) cfg.epsilon = std::atof(argv[5]);

    gfx::Window window({"34 GPU MDC", 1280, 720});

    gfx::Model model;
    if (!model.load(model_path)) {
        fprintf(stderr, "Error: failed to load %s\n", model_path);
        return 1;
    }
    printf("Loaded: %zu meshes\n", model.mesh_count());

    GpuResult gpu = run_gpu_pipeline(model, cfg);

    printf("\n=== CPU reference pipeline (example 31 machinery) ===\n");
    gfx::CoverageAtlas cpu_atlas(cfg);
    double cpu_ms = run_ms([&] {
        if (!cpu_atlas.build(model)) {
            fprintf(stderr, "Error: model has no geometry\n");
            std::exit(1);
        }
    });

    print_report(gpu, cpu_atlas, cpu_ms);

    run_render(window, gpu);
    return 0;
}

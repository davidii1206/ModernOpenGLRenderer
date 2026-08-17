// Example 36 — Micro Rendering: Stage A+B
//
// Stage A: G-buffer geometry pass (position, normal, albedo, emissive, depth)
//          with fullscreen display and ImGui visualization.
// Stage B: Offline best-candidate point sampling (Mitchell 1991) + complete
//          binary tree hierarchy (recursive median split, bounding spheres,
//          normal cones).  Debug point-cloud overlay.

#include <gl/gl.hpp>
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

out vec3 v_color;

void main() {
    gl_Position = u_view_proj * vec4(pgeom[gl_VertexID].xyz, 1.0);
    gl_PointSize = u_point_size;

    if (u_color_mode == 0)      v_color = palb[gl_VertexID].rgb;
    else if (u_color_mode == 1) v_color = pemit[gl_VertexID].rgb;
    else if (u_color_mode == 2) v_color = pnrm[gl_VertexID].rgb * 0.5 + 0.5;
    else if (u_color_mode == 3) v_color = pgeom[gl_VertexID].xyz;
    else                        v_color = rad[gl_VertexID].rgb;
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

layout(std430, binding = 0) writeonly buffer PGeom  { vec4 pgeom[];  };
layout(std430, binding = 1) writeonly buffer PNrm   { vec4 pnrm[];   };
layout(std430, binding = 4) readonly buffer TriBuf   { GpuTri tris[];  };
layout(std430, binding = 5) readonly buffer LeafSrcB { LeafSrc leaves[]; };
layout(std430, binding = 6) writeonly buffer Sphere  { vec4 sphere[];  };
layout(std430, binding = 7) writeonly buffer Cone    { vec4 cone[];    };

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
}
)";

const char* tree_refit_cs = R"(
#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 6) buffer Sphere { vec4 sphere[]; };
layout(std430, binding = 7) buffer Cone   { vec4 cone[];   };

uniform uint u_count;      // nodes at this level
uniform uint u_level_start; // first node index at this level

const float PI = 3.14159265358979;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= u_count) return;

    uint node = u_level_start + idx;
    uint left  = 2 * node + 1;
    uint right = 2 * node + 2;

    // --- Bounding sphere merge (Ritter) ---
    vec3 c1 = sphere[left].xyz,  c2 = sphere[right].xyz;
    float r1 = sphere[left].w,  r2 = sphere[right].w;
    vec3 d = c2 - c1;
    float dist = length(d);
    vec3 C; float R;
    if (dist < 1e-6) {
        C = (r1 >= r2) ? c1 : c2;
        R = max(r1, r2);
    } else if (dist + r1 <= r2) {
        C = c2; R = r2;
    } else if (dist + r2 <= r1) {
        C = c1; R = r1;
    } else {
        R = (dist + r1 + r2) * 0.5;
        C = c1 + d * ((R - r1) / dist);
    }
    sphere[node] = vec4(C, R);

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
// Stage D — Direct lighting (emissive point sources) + radiance pull-up
// ===========================================================================

const char* direct_lighting_cs = R"(
#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PGeom     { vec4 pgeom[];    };
layout(std430, binding = 1) readonly buffer PNrm      { vec4 pnrm[];     };
layout(std430, binding = 3) readonly buffer PEmit      { vec4 pemit[];    };
layout(std430, binding = 8) writeonly buffer Radiance  { vec4 radiance[]; };
layout(std430, binding = 9) readonly buffer Emitters   { uint emitters[]; };

uniform uint u_num_leaves;
uniform uint u_num_emitters;
uniform float u_leaf_area;

void main() {
    uint recv = gl_GlobalInvocationID.x;
    if (recv >= u_num_leaves) return;

    vec3 pos = pgeom[recv].xyz;
    vec3 nrm = normalize(pnrm[recv].xyz);
    vec3 emissive = pemit[recv].rgb;

    vec3 incoming = vec3(0.0);
    for (uint i = 0; i < u_num_emitters; i++) {
        uint e = emitters[i];

        vec3 e_pos = pgeom[e].xyz;
        vec3 e_nrm = normalize(pnrm[e].xyz);
        vec3 e_emit = pemit[e].rgb;

        vec3 dir = e_pos - pos;
        float dist2 = max(dot(dir, dir), 1e-4);
        float dist = sqrt(dist2);
        vec3 wi = dir / dist;

        float cos_recv = max(dot(nrm, wi), 0.0);
        float cos_emit = max(dot(e_nrm, -wi), 0.0);

        incoming += e_emit * cos_emit * cos_recv / dist2 * u_leaf_area;
    }

    radiance[recv] = vec4(emissive + incoming, u_leaf_area);
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
    if (w.key_down(gfx::Key::a)) vel += right;
    if (w.key_down(gfx::Key::d)) vel -= right;
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

    // Bottom-up refit
    for (int node = N - 2; node >= 0; --node) {
        int l = 2 * node + 1;
        int r = 2 * node + 2;

        // Bounding sphere merge (Ritter's algorithm)
        glm::vec3 c1(ph.sphere[l].x, ph.sphere[l].y, ph.sphere[l].z);
        glm::vec3 c2(ph.sphere[r].x, ph.sphere[r].y, ph.sphere[r].z);
        float r1 = ph.sphere[l].w, r2 = ph.sphere[r].w;
        glm::vec3 d = c2 - c1;
        float dist = glm::length(d);
        glm::vec3 C;
        float R;
        if (dist < 1e-6f) {
            C = (r1 >= r2) ? c1 : c2;
            R = std::max(r1, r2);
        } else if (dist + r1 <= r2) {
            C = c2; R = r2;
        } else if (dist + r2 <= r1) {
            C = c1; R = r1;
        } else {
            R = (dist + r1 + r2) * 0.5f;
            C = c1 + d * ((R - r1) / dist);
        }
        ph.sphere[node] = glm::vec4(C, R);

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
    gfx::Window window({"36 Micro Rendering — Stage A+B+C+D", 1600, 900});
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
    gl::Program direct_light_prog = make_compute(direct_lighting_cs);
    gl::Program radiance_pull_prog = make_compute(radiance_pullup_cs);
    gl::Program sphere_prog      = make_program(sphere_vs, sphere_fs);

    if (!gbuf_prog.linked() || !display_prog.linked() || !pc_prog.linked() ||
        !leaf_update_prog.linked() || !tree_refit_prog.linked() ||
        !direct_light_prog.linked() || !radiance_pull_prog.linked() ||
        !sphere_prog.linked()) {
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
    radiance_buf.data(nullptr, size_t(N) * sizeof(glm::vec4));  // filled by compute
    radiance_buf.bind_base(8);

    float leaf_area = (emitter_indices.empty() || N == 0) ? 0.0f : ph.total_area / float(N);

    // --- Empty VAO for point cloud (shader reads from SSBO via gl_VertexID) ---
    gl::VertexArray pc_vao;

    // --- Stage C: wireframe bounding sphere debug mesh ---
    SphereMesh sphere_mesh = create_wireframe_sphere(8, 16);
    setup_sphere_vao(sphere_mesh, sphere_buf);

    // --- G-buffer ---
    GBuffer gbuf;
    create_gbuffer(gbuf, window.framebuffer_width(), window.framebuffer_height());

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
    float refit_ms = 0.0f;
    float max_pos_err = 0.0f;
    float max_cone_err = 0.0f;
    glm::vec4 sphere_color(0.0f, 1.0f, 0.0f, 0.3f);
    int sphere_lod = 0;  // 0=all, 1=leaves only, 2=interior only

    double last = window.time();

    while (!window.should_close()) {
        double now = window.time();
        float dt = float(now - last);
        last = now;

        window.poll_events();
        camera_control(window, cam, dt, !gui.wants_mouse(), captured);
        cam.set_aspect(float(window.framebuffer_width()) / float(window.framebuffer_height()));

        int fw = window.framebuffer_width(), fh = window.framebuffer_height();
        if (fw != gbuf.w || fh != gbuf.h)
            create_gbuffer(gbuf, fw, fh);

        glm::mat4 vp = cam.view_projection();

        // ===================================================================
        // 1. Geometry pass
        // ===================================================================
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

    // ===================================================================
        // 1b. Stage C — GPU refit: leaf update + bottom-up tree refit
        // ===================================================================
        if (run_refit) {
            auto t0 = std::chrono::steady_clock::now();

            // Bind compute-writeable buffers
            pgeom_buf.bind_base(0);
            pnrm_buf.bind_base(1);
            palb_buf.bind_base(2);
            pemit_buf.bind_base(3);
            tri_buf.bind_base(4);
            leaf_src_buf.bind_base(5);
            sphere_buf.bind_base(6);
            cone_buf.bind_base(7);

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

            auto t1 = std::chrono::steady_clock::now();
            refit_ms = float(std::chrono::duration<double, std::milli>(t1 - t0).count());

            // --- Stage D: Direct lighting from emissive leaves ---
            radiance_buf.bind_base(8);
            emitters_buf.bind_base(9);

            direct_light_prog.use();
            loc_c = direct_light_prog.uniform_location("u_num_leaves");
            if (loc_c >= 0) glProgramUniform1ui(direct_light_prog.handle(), loc_c, GLuint(N));
            loc_c = direct_light_prog.uniform_location("u_num_emitters");
            if (loc_c >= 0) glProgramUniform1ui(direct_light_prog.handle(), loc_c, GLuint(emitter_indices.size()));
            loc_c = direct_light_prog.uniform_location("u_leaf_area");
            if (loc_c >= 0) glProgramUniform1f(direct_light_prog.handle(), loc_c, leaf_area);
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
        }

    // ===================================================================
        // 2. Display pass — fullscreen triangle showing selected G-buffer target
        // ===================================================================
        gl::disable(GL_DEPTH_TEST);
        gl::viewport(0, 0, fw, fh);
        gl::clear_color(0.02f, 0.02f, 0.03f, 1.0f);
        gl::clear(GL_COLOR_BUFFER_BIT);

        display_prog.use();
        fsq_vao.bind();
        gl::Texture* targets[] = { &gbuf.albedo, &gbuf.normal, &gbuf.position,
                                    &gbuf.emissive, &gbuf.depth };
        targets[view_mode]->bind(0);
        loc = display_prog.uniform_location("u_tex");      if (loc >= 0) display_prog.uniform1i(loc, 0);
        loc = display_prog.uniform_location("u_mode");     if (loc >= 0) display_prog.uniform1i(loc, view_mode + 1);
        loc = display_prog.uniform_location("u_far");      if (loc >= 0) display_prog.uniform1f(loc, far_plane);
        loc = display_prog.uniform_location("u_exposure"); if (loc >= 0) display_prog.uniform1f(loc, exposure);
        loc = display_prog.uniform_location("u_gamma");    if (loc >= 0) display_prog.uniform1f(loc, gamma);
        gl::draw_arrays(GL_TRIANGLES, 0, 3);

        // ===================================================================
        // 2b. Point cloud debug overlay
        // ===================================================================
        if (show_points) {
            pc_prog.use();
            loc = pc_prog.uniform_location("u_view_proj");
            if (loc >= 0) pc_prog.uniform_matrix4fv(loc, glm::value_ptr(vp));
            loc = pc_prog.uniform_location("u_point_size");
            if (loc >= 0) pc_prog.uniform1f(loc, point_size);
            loc = pc_prog.uniform_location("u_color_mode");
            if (loc >= 0) pc_prog.uniform1i(loc, pc_color_mode);

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
        // 3. ImGui
        // ===================================================================
        gui.begin_frame();
        {
            ImGui::Begin("Stage A+B+C+D — Micro Rendering");
            ImGui::Text("FPS: %.1f  Frame: %.2f ms", 1.0f / std::max(dt, 1e-6f), dt * 1000.0f);
            ImGui::Text("Resolution: %d x %d", gbuf.w, gbuf.h);
            ImGui::Separator();

            // G-buffer view
            ImGui::Combo("G-Buffer View", &view_mode,
                         "Albedo\0Normal\0Position\0Emissive\0Depth\0");
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
            ImGui::Text("  %.2f ms", refit_ms);
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
            ImGui::Text("  Switch to 'Radiance' color mode to visualize");

            ImGui::End();
        }
        gui.render();

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}

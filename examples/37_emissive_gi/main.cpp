// Example 37 — Point-Sampled Emissive Lighting
//
// Micro-rendering for scalable, parallel final gathering (Ritschel et al. 2009)
//   x  Ray Tracing Point Sampled Geometry (Schaufler & Jensen 2000)
// in one emissive-forward pipeline: the ONLY forward light is a directional light;
// everything else comes from emissive materials gathered through a BVH8 point octree.
//
// This file = scaffolding (S1 + S2): window, camera, hot-reload shaders, multi-model
// load/toggle, forward G-buffer, display pass. Later stages add the surfel octree
// (S3/S4), the micro-render gather (S5/S6), S-J shadow/specular rays (S7/S8) and
// bilateral upsampling + UI (S9). See implementation.md.

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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// S3 — surfel sampling + BVH8 octree (see implementation.md §3/§7)
// ---------------------------------------------------------------------------

constexpr float kPi        = 3.14159265358979f;
constexpr int   kBEST_K    = 24;    // Mitchell best-candidate candidates
constexpr float kCOVERAGE  = 1.4f;  // leaf radius = coverage * sqrt(area/(N*pi))
constexpr uint32_t kSEED   = 42;
constexpr int   kBudgetOptions[] = {4096, 32768, 262144};
constexpr int   kBudgetCount = 3;

struct Tri {
    glm::vec3 p[3];
    glm::vec3 n[3];
    float area = 0.0f;
    glm::vec3 alb{0.0f};
    glm::vec3 emi{0.0f};
};

struct SamplePoint {
    glm::vec3 pos{0.0f};
    glm::vec3 nrm{0.0f};
    glm::vec3 alb{0.0f};
    glm::vec3 emi{0.0f};
    int tri_idx = 0;
    float u = 0.0f, v = 0.0f;   // barycentric (for S4 refit)
};

// Complete 8-ary tree: N = 8^depth leaves, children of node i at 8i+1..8i+8.
struct Octree8 {
    int N = 0;
    int depth = 0;
    int interior = 0;       // (N-1)/7 interior nodes, [0, interior)
    int total_nodes = 0;    // (8N-1)/7, leaves at [interior, total)
    float leaf_radius = 0.0f;
    float total_area = 0.0f;
    std::vector<SamplePoint> pts;    // in leaf order
    std::vector<glm::vec4> sphere;   // total_nodes: (center, radius)
    std::vector<glm::vec4> cone;     // total_nodes: (axis, cos half-angle)
};

// Packed records (S3b). 16-byte leaf, 64-byte interior bounds, 8-byte cone.
struct PackedLeaf   { uint32_t xy; uint32_t zw; uint32_t alb; uint32_t nrm; };
struct PackedChild  { uint16_t pos[3]; uint16_t radius; };
struct PackedBounds { PackedChild child[8]; };              // interior nodes
struct PackedCone   { uint16_t axis[2]; uint16_t half; uint16_t pad; };  // 8 B (uvec2)
struct PackedSphere { uint32_t x, y; };                     // per-node packed sphere (8 B)

// S4: GPU triangle + leaf-source records (match compute shader layouts).
struct GpuTri {
    glm::vec4 pos[3];
    glm::vec4 nrm[3];
};
struct LeafSrc {
    uint32_t tri_idx;
    float u, v, pad;
};

// P0: double-buffered GPU timer for per-pass profiling.
struct GpuTimer {
    const char* name;
    gl::Query q{gl::QueryType::time_elapsed};
    gl::Query q_prev{gl::QueryType::time_elapsed};
    bool ran = false;
    double ms = 0.0;
    explicit GpuTimer(const char* n) : name(n) {}
    void begin() { q.begin(); }
    void end() { q.end(); std::swap(q, q_prev); ran = true; }
    double readback() {
        if (ran) { ms = double(q_prev.result()) * 1e-6; ran = false; }
        return ms;
    }
};

// ---------------------------------------------------------------------------
// Forward G-buffer: position / normal / albedo / emissive / linear depth
// ---------------------------------------------------------------------------

struct FrameTargets {
    int w = 0, h = 0;
    gl::Texture position{gl::TextureType::tex_2d};
    gl::Texture normal{gl::TextureType::tex_2d};
    gl::Texture albedo{gl::TextureType::tex_2d};
    gl::Texture emissive{gl::TextureType::tex_2d};
    gl::Texture depth{gl::TextureType::tex_2d};
    gl::Renderbuffer depth_rbo;
    gl::Framebuffer fbo;
};

void create_targets(FrameTargets& t, int w, int h) {
    t.w = w;
    t.h = h;

    auto tex2 = [&](gl::Texture& tex, GLenum internal) {
        tex = gl::Texture{gl::TextureType::tex_2d};   // fresh storage (immutable re-spec is invalid)
        tex.image_2d(0, internal, w, h,
                     internal == GL_R32F ? GL_RED : GL_RGBA,
                     GL_FLOAT, nullptr, 1);
        tex.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        tex.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        tex.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        tex.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };

    tex2(t.position, GL_RGBA16F);
    tex2(t.normal,   GL_RGBA16F);
    tex2(t.albedo,   GL_RGBA8);
    tex2(t.emissive, GL_RGBA16F);
    tex2(t.depth,    GL_R32F);

    t.depth_rbo = gl::Renderbuffer{};   // fresh storage (immutable re-spec is invalid)
    t.depth_rbo.storage(GL_DEPTH_COMPONENT24, w, h);

    t.fbo.bind();
    t.fbo.attach_texture(GL_COLOR_ATTACHMENT0, t.position);
    t.fbo.attach_texture(GL_COLOR_ATTACHMENT1, t.normal);
    t.fbo.attach_texture(GL_COLOR_ATTACHMENT2, t.albedo);
    t.fbo.attach_texture(GL_COLOR_ATTACHMENT3, t.emissive);
    t.fbo.attach_texture(GL_COLOR_ATTACHMENT4, t.depth);
    t.fbo.attach_renderbuffer(GL_DEPTH_ATTACHMENT, t.depth_rbo);
    GLenum bufs[5] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
                      GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3,
                      GL_COLOR_ATTACHMENT4};
    glDrawBuffers(5, bufs);
    if (!t.fbo.check())
        gllib::log(gllib::LogLevel::error, "G-buffer framebuffer incomplete");
    gl::Framebuffer::unbind(gl::FramebufferType::both);
}

// ---------------------------------------------------------------------------
// Orbit camera control (right-drag to orbit, scroll to zoom, WASD to move)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Model handling: available scenes, per-model transform
// ---------------------------------------------------------------------------

struct ModelEntry {
    const char* path;
    const char* label;
};

static const ModelEntry kModels[] = {
    {"CornellBoxOriginal.glb", "Cornell Box"},
    {"sponza.glb",             "Sponza"},
};

// Uniform-scale + center the model so it fits a target size around the origin.
// CornellBox is axis-aligned with +X up; rotate it so +Y is up.
// Bounds are computed from actual vertex positions (per-mesh bounding spheres are
// not reliable across glTF node transforms), so the result is robust for any scene.
glm::mat4 model_transform(const gfx::Model& model, std::string_view path, float target_size) {
    glm::mat4 pre = glm::mat4(1.0f);
    if (path.find("CornellBox") != std::string_view::npos)
        pre = glm::rotate(pre, glm::radians(90.0f), glm::vec3(1, 0, 0));

    glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);
    for (size_t i = 0; i < model.mesh_count(); ++i) {
        const auto& mesh = model.mesh(i);
        for (const auto& v : mesh.vertices()) {
            glm::vec4 p = pre * glm::vec4(v.position[0], v.position[1], v.position[2], 1.0f);
            for (int j = 0; j < 3; ++j) {
                lo[j] = std::min(lo[j], p[j]);
                hi[j] = std::max(hi[j], p[j]);
            }
        }
    }

    glm::vec3 center = (lo + hi) * 0.5f;
    glm::vec3 ext = hi - lo;
    float size = std::max({ext.x, ext.y, ext.z, 1e-6f});
    float scale = target_size / size;

    // final = scale * (pre * v) - scale * center  -> geometry centered at origin
    glm::mat4 m = glm::translate(glm::mat4(1.0f), -center * scale) *
                  pre * glm::scale(glm::mat4(1.0f), glm::vec3(scale));

    gllib::logf(gllib::LogLevel::info,
                "  model bounds(rotated): (%.2f %.2f %.2f)-(%.2f %.2f %.2f) scale=%.5f",
                lo.x, lo.y, lo.z, hi.x, hi.y, hi.z, scale);
    return m;
}

// ---------------------------------------------------------------------------
// S3 — triangle extraction (bakes model_mat into world space)
// ---------------------------------------------------------------------------

std::vector<Tri> extract_triangles(const gfx::Model& model, const glm::mat4& model_mat) {
    std::vector<Tri> tris;
    glm::mat3 normal_mat = glm::transpose(glm::inverse(glm::mat3(model_mat)));

    for (size_t mi = 0; mi < model.mesh_count(); ++mi) {
        const gfx::Mesh& mesh = model.mesh(int(mi));
        int mati = model.mesh_material(int(mi));
        const auto& mat = model.material_info(size_t(mati >= 0 ? mati : 0));
        glm::vec3 alb(mat.base_color_factor[0], mat.base_color_factor[1], mat.base_color_factor[2]);
        glm::vec3 emi(mat.emissive_factor[0], mat.emissive_factor[1], mat.emissive_factor[2]);
        const auto& vs = mesh.vertices();
        const auto& is = mesh.indices();

        auto add_tri = [&](const gfx::Vertex& a, const gfx::Vertex& b, const gfx::Vertex& c) {
            Tri t;
            glm::vec3 pa(a.position[0], a.position[1], a.position[2]);
            glm::vec3 pb(b.position[0], b.position[1], b.position[2]);
            glm::vec3 pc(c.position[0], c.position[1], c.position[2]);
            t.p[0] = glm::vec3(model_mat * glm::vec4(pa, 1.0f));
            t.p[1] = glm::vec3(model_mat * glm::vec4(pb, 1.0f));
            t.p[2] = glm::vec3(model_mat * glm::vec4(pc, 1.0f));
            glm::vec3 na = normal_mat * glm::vec3(a.normal[0], a.normal[1], a.normal[2]);
            glm::vec3 nb = normal_mat * glm::vec3(b.normal[0], b.normal[1], b.normal[2]);
            glm::vec3 nc = normal_mat * glm::vec3(c.normal[0], c.normal[1], c.normal[2]);
            t.n[0] = glm::normalize(na);
            t.n[1] = glm::normalize(nb);
            t.n[2] = glm::normalize(nc);
            t.area = 0.5f * glm::length(glm::cross(t.p[1] - t.p[0], t.p[2] - t.p[0]));
            if (t.area < 1e-9f) return;
            t.alb = alb;
            t.emi = emi;
            tris.push_back(t);
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

// ---------------------------------------------------------------------------
// S3 — best-candidate sampling (Mitchell 1991), area-weighted
// ---------------------------------------------------------------------------

SamplePoint sample_on_tri(const Tri& tri, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float su = dist(rng), sv = dist(rng);
    if (su + sv > 1.0f) { su = 1.0f - su; sv = 1.0f - sv; }
    float sw = 1.0f - su - sv;
    SamplePoint s;
    s.pos = su * tri.p[0] + sv * tri.p[1] + sw * tri.p[2];
    s.nrm = glm::normalize(su * tri.n[0] + sv * tri.n[1] + sw * tri.n[2]);
    s.alb = tri.alb;
    s.emi = tri.emi;
    s.u = su;
    s.v = sv;
    return s;
}

std::vector<SamplePoint> sample_points_mitchell(
    const std::vector<Tri>& tris, float total_area, int N, int K)
{
    std::vector<float> cdf(tris.size());
    float acc = 0.0f;
    for (size_t i = 0; i < tris.size(); ++i) { acc += tris[i].area; cdf[i] = acc; }

    // Scene AABB for the acceleration grid.
    glm::vec3 mn(1e30f), mx(-1e30f);
    for (const auto& t : tris)
        for (int k = 0; k < 3; ++k)
            for (int c = 0; c < 3; ++c) {
                mn[c] = std::min(mn[c], t.p[k][c]);
                mx[c] = std::max(mx[c], t.p[k][c]);
            }
    float s = std::max(std::sqrt(total_area / float(N)), 1e-6f);  // expected spacing

    // Uniform grid over the scene; cell size = spacing. Only occupied cells stored.
    auto cell_of = [&](const glm::vec3& p) -> glm::ivec3 {
        return glm::ivec3(glm::floor((p - mn) / s));
    };
    auto cell_key = [](const glm::ivec3& c) -> uint64_t {
        return (uint64_t(uint32_t(c.x) & 0x1FFFFF) << 42) |
               (uint64_t(uint32_t(c.y) & 0x1FFFFF) << 21) |
                uint64_t(uint32_t(c.z) & 0x1FFFFF);
    };
    std::unordered_map<uint64_t, std::vector<int>> grid;

    std::mt19937 rng(kSEED);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::uniform_int_distribution<int> tri_pick(0, int(tris.size()) - 1);

    std::vector<SamplePoint> accepted;
    accepted.reserve(N);

    auto add_point = [&](int idx) {
        grid[cell_key(cell_of(accepted[idx].pos))].push_back(idx);
    };

    // Nearest accepted-point squared distance to p (expanding ring search).
    // stop_at: we only need to know whether any accepted point is closer than
    // this (the best candidate distance found so far), so we can early-out.
    // Ring depth is capped (kMaxRing*s): beyond that we treat the neighborhood
    // as empty — a standard, quality-preserving Mitchell approximation.
    constexpr int kMaxRing = 6;
    auto min_dist_to = [&](const glm::vec3& p, float stop_at, float& d2) -> bool {
        glm::ivec3 c0 = cell_of(p);
        d2 = 1e30f;
        for (int R = 0; R <= kMaxRing; ++R) {
            bool any = false;
            for (int dx = -R; dx <= R; ++dx)
                for (int dy = -R; dy <= R; ++dy)
                    for (int dz = -R; dz <= R; ++dz) {
                        if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != R) continue;
                        auto it = grid.find(cell_key(c0 + glm::ivec3(dx, dy, dz)));
                        if (it == grid.end()) continue;
                        any = true;
                        for (int idx : it->second) {
                            float t = glm::dot(p - accepted[idx].pos, p - accepted[idx].pos);
                            if (t < d2) { d2 = t; if (t == 0.0f) return true; }
                        }
                    }
            // Unchecked points are in rings >= R+1, at least R*s away.
            float bound = std::max(stop_at, float(R * R) * s * s);
            if (any && d2 <= bound) break;
        }
        return d2 < 1e29f;
    };

    // First sample.
    {
        int ti = tri_pick(rng);
        SamplePoint sp = sample_on_tri(tris[ti], rng);
        sp.tri_idx = ti;
        accepted.push_back(sp);
        add_point(0);
    }

    for (int i = 1; i < N; ++i) {
        float best_dist = -1.0f;
        SamplePoint best_sp{};
        for (int k = 0; k < K; ++k) {
            float r = unit(rng) * total_area;
            int ti = int(std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin());
            ti = std::clamp(ti, 0, int(tris.size()) - 1);
            SamplePoint cand = sample_on_tri(tris[ti], rng);
            cand.tri_idx = ti;

            float min_d;
            if (!min_dist_to(cand.pos, best_dist, min_d)) min_d = 1e30f;
            if (min_d > best_dist) { best_dist = min_d; best_sp = cand; }
        }
        accepted.push_back(best_sp);
        add_point(i);
    }
    return accepted;
}

// ---------------------------------------------------------------------------
// S3 — complete 8-ary tree build (balanced median split, bottom-up refit)
// ---------------------------------------------------------------------------

void merge_normal_cones(const glm::vec3& a1, float w1,
                        const glm::vec3& a2, float w2,
                        glm::vec3& axis, float& cw) {
    float b1 = std::acos(std::clamp(w1, -1.0f, 1.0f));
    float b2 = std::acos(std::clamp(w2, -1.0f, 1.0f));
    float ang = std::acos(std::clamp(glm::dot(a1, a2), -1.0f, 1.0f));
    if (ang + b1 + b2 >= kPi - 1e-5f || glm::length(a1 + a2) < 1e-4f) {
        axis = a1; cw = -1.0f;
    } else {
        axis = glm::normalize(a1 + a2);
        cw = std::cos((ang + b1 + b2) * 0.5f);
    }
}

Octree8 build_octree8(std::vector<SamplePoint> points, float total_area, float coverage) {
    int N = int(points.size());
    int depth = 0;
    for (int t = N; t > 1; t /= 8) ++depth;

    Octree8 o;
    o.N = N;
    o.depth = depth;
    o.interior = (N - 1) / 7;
    o.total_nodes = (8 * N - 1) / 7;
    o.total_area = total_area;
    o.leaf_radius = coverage * std::sqrt(total_area / (float(N) * kPi));

    // 1. Determine leaf order via recursive 8-way median split (largest axis).
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::vector<glm::vec3> P(N);
    for (int i = 0; i < N; ++i) P[i] = points[i].pos;

    std::function<void(int, int)> rec = [&](int lo, int hi) {
        int count = hi - lo;
        if (count <= 1) return;
        glm::vec3 bmin(1e30f), bmax(-1e30f);
        for (int k = lo; k < hi; ++k) {
            bmin = glm::min(bmin, P[order[k]]);
            bmax = glm::max(bmax, P[order[k]]);
        }
        glm::vec3 ext = bmax - bmin;
        int axis = (ext.y > ext.x && ext.y > ext.z) ? 1
                 : ((ext.z > ext.x && ext.z > ext.y) ? 2 : 0);
        for (int b = 1; b < 8; ++b) {
            int boundary = lo + (count * b) / 8;
            std::nth_element(order.begin() + lo, order.begin() + boundary, order.begin() + hi,
                [&](int a, int c) { return P[a][axis] < P[c][axis]; });
        }
        int seg = count / 8;
        for (int b = 0; b < 8; ++b)
            rec(lo + b * seg, lo + (b + 1) * seg);
    };
    rec(0, N);

    std::vector<SamplePoint> ordered(N);
    for (int i = 0; i < N; ++i) ordered[i] = points[order[i]];
    o.pts = std::move(ordered);

    // 2. Leaves.
    o.sphere.assign(o.total_nodes, glm::vec4(0.0f));
    o.cone.assign(o.total_nodes, glm::vec4(0.0f));
    for (int i = 0; i < N; ++i) {
        int node = o.interior + i;
        o.sphere[node] = glm::vec4(o.pts[i].pos, o.leaf_radius);
        o.cone[node] = glm::vec4(o.pts[i].nrm, 1.0f);
    }

    // 3. Bottom-up refit: Ritter bounding sphere over the 8 child spheres +
    //    merged normal cone (mirrors refit_bottom_up.comp exactly).
    for (int node = o.interior - 1; node >= 0; --node) {
        glm::vec3 c = glm::vec3(o.sphere[8 * node + 1]);
        float r = o.sphere[8 * node + 1].w;
        glm::vec3 axis(0.0f); float cw = 1.0f; bool first = true;
        for (int ch = 0; ch < 8; ++ch) {
            int child = 8 * node + 1 + ch;
            const glm::vec3& cc = glm::vec3(o.sphere[child]);
            float cr = o.sphere[child].w;
            float d = glm::length(cc - c);
            if (d + cr > r) {
                float nr = (r + d + cr) * 0.5f;
                c = (d > 1e-12f) ? c + (cc - c) * (nr - r) / d : c;
                r = nr;
            }
            const glm::vec3& ca = glm::vec3(o.cone[child]);
            if (first) { axis = ca; cw = o.cone[child].w; first = false; }
            else merge_normal_cones(axis, cw, ca, o.cone[child].w, axis, cw);
        }
        o.sphere[node] = glm::vec4(c, r);
        o.cone[node] = glm::vec4(axis, cw);
    }
    return o;
}

// ---------------------------------------------------------------------------
// S3b — packing helpers (quantized pos, octahedral normals, u8 albedo)
// ---------------------------------------------------------------------------

glm::vec2 oct_encode(const glm::vec3& n) {
    glm::vec3 p = n / (std::abs(n.x) + std::abs(n.y) + std::abs(n.z));
    if (p.z < 0.0f) {
        float tx = p.x, ty = p.y;
        p.x = (1.0f - std::abs(ty)) * (tx >= 0.0f ? 1.0f : -1.0f);
        p.y = (1.0f - std::abs(tx)) * (ty >= 0.0f ? 1.0f : -1.0f);
    }
    return glm::vec2(p.x, p.y);
}

struct PackParams {
    glm::vec3 scene_min{0.0f};
    glm::vec3 scene_size{1.0f};
    float radius_scale = 1.0f;
};

uint16_t q_u16(float t) { return uint16_t(std::clamp(int(std::lround(t * 65535.0f)), 0, 65535)); }

void pack_leaf(const SamplePoint& s, float radius, const PackParams& pp, PackedLeaf& out) {
    glm::vec3 t = glm::clamp((s.pos - pp.scene_min) / pp.scene_size, 0.0f, 1.0f);
    uint16_t px = q_u16(t.x), py = q_u16(t.y), pz = q_u16(t.z);
    uint16_t r  = q_u16(std::clamp(radius / pp.radius_scale, 0.0f, 1.0f));
    glm::vec2 o = oct_encode(s.nrm);
    int16_t ox = int16_t(std::clamp(int(std::lround(o.x * 32767.0f)), -32768, 32767));
    int16_t oy = int16_t(std::clamp(int(std::lround(o.y * 32767.0f)), -32768, 32767));
    uint8_t ar = uint8_t(std::clamp(int(std::lround(s.alb.r * 255.0f)), 0, 255));
    uint8_t ag = uint8_t(std::clamp(int(std::lround(s.alb.g * 255.0f)), 0, 255));
    uint8_t ab = uint8_t(std::clamp(int(std::lround(s.alb.b * 255.0f)), 0, 255));
    out.xy = (uint32_t(px) << 16) | uint32_t(r);
    out.zw = (uint32_t(py) << 16) | uint32_t(pz);
    out.alb = (uint32_t(ar) << 16) | (uint32_t(ag) << 8) | uint32_t(ab);
    out.nrm = (uint32_t(uint16_t(ox)) << 16) | uint32_t(uint16_t(oy));
}

void pack_bounds(const glm::vec3* centers, const float* radii, const PackParams& pp, PackedBounds& out) {
    for (int c = 0; c < 8; ++c) {
        glm::vec3 t = glm::clamp((centers[c] - pp.scene_min) / pp.scene_size, 0.0f, 1.0f);
        out.child[c].pos[0] = q_u16(t.x);
        out.child[c].pos[1] = q_u16(t.y);
        out.child[c].pos[2] = q_u16(t.z);
        out.child[c].radius = q_u16(std::clamp(radii[c] / pp.radius_scale, 0.0f, 1.0f));
    }
}

void pack_cone(const glm::vec3& axis, float cw, PackedCone& out) {
    glm::vec2 o = oct_encode(glm::normalize(axis));
    int16_t ox = int16_t(std::clamp(int(std::lround(o.x * 32767.0f)), -32768, 32767));
    int16_t oy = int16_t(std::clamp(int(std::lround(o.y * 32767.0f)), -32768, 32767));
    out.axis[0] = uint16_t(ox);
    out.axis[1] = uint16_t(oy);
    out.half = uint16_t(std::clamp(int(std::lround((cw + 1.0f) * 0.5f * 255.0f)), 0, 255));
    out.pad = 0;
}

// ---------------------------------------------------------------------------
// S3 — binary cache with explicit invalidation
// ---------------------------------------------------------------------------

constexpr uint32_t kCacheMagic   = 0x53374D45;  // "S7ME"
constexpr uint32_t kCacheVersion = 3;   // bump when sampling/extraction changes

uint64_t file_stamp(const std::string& path, uint64_t* size_out) {
    std::error_code ec;
    auto ftime = std::filesystem::last_write_time(path, ec);
    uint64_t stamp = ec ? 0 : uint64_t(ftime.time_since_epoch().count());
    uint64_t size = ec ? 0 : uint64_t(std::filesystem::file_size(path, ec));
    if (size_out) *size_out = size;
    return (stamp ^ (size * 0x9E3779B97F4A7C15ull));
}

uint64_t material_hash(const gfx::Model& model) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < model.material_count(); ++i) {
        const auto& m = model.material_info(i);
        for (int k = 0; k < 4; ++k) h = (h ^ uint64_t(std::bit_cast<uint32_t>(m.base_color_factor[k]))) * 1099511628211ull;
        for (int k = 0; k < 3; ++k) h = (h ^ uint64_t(std::bit_cast<uint32_t>(m.emissive_factor[k]))) * 1099511628211ull;
    }
    return h;
}

struct CacheKey {
    uint64_t model_stamp = 0;
    uint64_t mat_hash = 0;
    uint64_t transform_hash = 0;   // model_mat baked into world-space samples
    uint32_t N = 0;
    uint32_t best_k = kBEST_K;
    float coverage = kCOVERAGE;
    uint32_t seed = kSEED;
};

uint64_t matrix_hash(const glm::mat4& m) {
    uint64_t h = 1469598103934665603ull;
    const float* p = glm::value_ptr(m);
    for (int i = 0; i < 16; ++i)
        h = (h ^ uint64_t(std::bit_cast<uint32_t>(p[i]))) * 1099511628211ull;
    return h;
}

std::string cache_path_for(const std::string& model_path, int N) {
    std::filesystem::path p(model_path);
    return "cache/37_" + p.stem().string() + "_N" + std::to_string(N) + ".bin";
}

bool save_cache(const std::string& path, const CacheKey& key, const Octree8& o,
                const std::vector<Tri>& tris) {
    std::filesystem::create_directories("cache");
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    auto w32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto w64 = [&](uint64_t v) { f.write(reinterpret_cast<const char*>(&v), 8); };
    auto wf  = [&](float v)    { f.write(reinterpret_cast<const char*>(&v), 4); };

    w32(kCacheMagic);
    w32(kCacheVersion);
    w64(key.model_stamp);
    w64(key.mat_hash);
    w64(key.transform_hash);
    w32(key.N);
    w32(key.best_k);
    wf(key.coverage);
    w32(key.seed);
    wf(o.leaf_radius);
    wf(o.total_area);
    w32(uint32_t(tris.size()));

    f.write(reinterpret_cast<const char*>(o.pts.data()), o.pts.size() * sizeof(SamplePoint));
    f.write(reinterpret_cast<const char*>(o.sphere.data()), o.sphere.size() * sizeof(glm::vec4));
    f.write(reinterpret_cast<const char*>(o.cone.data()), o.cone.size() * sizeof(glm::vec4));
    f.write(reinterpret_cast<const char*>(tris.data()), tris.size() * sizeof(Tri));
    return f.good();
}

bool load_cache(const std::string& path, const CacheKey& key, Octree8& o, std::vector<Tri>& tris) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    auto r32 = [&]() -> uint32_t { uint32_t v = 0; f.read(reinterpret_cast<char*>(&v), 4); return v; };
    auto r64 = [&]() -> uint64_t { uint64_t v = 0; f.read(reinterpret_cast<char*>(&v), 8); return v; };
    auto rf  = [&]() -> float    { float v = 0; f.read(reinterpret_cast<char*>(&v), 4); return v; };

    if (r32() != kCacheMagic) return false;
    if (r32() != kCacheVersion) return false;
    if (r64() != key.model_stamp) return false;
    if (r64() != key.mat_hash) return false;
    if (r64() != key.transform_hash) return false;
    if (r32() != key.N) return false;
    if (r32() != key.best_k) return false;
    if (rf() != key.coverage) return false;
    if (r32() != key.seed) return false;

    o.leaf_radius = rf();
    o.total_area = rf();
    int ntris = int(r32());

    o.N = int(key.N);
    o.interior = (o.N - 1) / 7;
    o.total_nodes = (8 * o.N - 1) / 7;
    o.depth = 0;
    for (int t = o.N; t > 1; t /= 8) ++o.depth;

    o.pts.resize(o.N);
    f.read(reinterpret_cast<char*>(o.pts.data()), o.N * sizeof(SamplePoint));
    o.sphere.resize(o.total_nodes);
    f.read(reinterpret_cast<char*>(o.sphere.data()), o.total_nodes * sizeof(glm::vec4));
    o.cone.resize(o.total_nodes);
    f.read(reinterpret_cast<char*>(o.cone.data()), o.total_nodes * sizeof(glm::vec4));
    tris.resize(ntris);
    f.read(reinterpret_cast<char*>(tris.data()), ntris * sizeof(Tri));
    return f.good();
}

// ---------------------------------------------------------------------------
// Unit sphere wireframe mesh for bounding-sphere debug overlay
// ---------------------------------------------------------------------------

struct SphereMesh {
    gl::VertexArray vao;
    gl::Buffer vbo, ebo;
    int index_count = 0;
};

SphereMesh create_wireframe_sphere(int lat_seg, int lon_seg) {
    SphereMesh sm;
    std::vector<glm::vec3> verts;
    std::vector<uint32_t> idx;

    verts.emplace_back(0.0f, 1.0f, 0.0f);
    for (int i = 1; i < lat_seg; ++i) {
        float phi = kPi * float(i) / float(lat_seg);
        float sp = std::sin(phi), cp = std::cos(phi);
        for (int j = 0; j < lon_seg; ++j) {
            float theta = 2.0f * kPi * float(j) / float(lon_seg);
            verts.emplace_back(sp * std::cos(theta), cp, sp * std::sin(theta));
        }
    }
    verts.emplace_back(0.0f, -1.0f, 0.0f);

    for (int j = 0; j < lon_seg; ++j) {
        idx.push_back(0);
        idx.push_back(1 + j);
        idx.push_back(1 + (j + 1) % lon_seg);
    }
    for (int i = 0; i < lat_seg - 2; ++i) {
        int row0 = 1 + i * lon_seg;
        int row1 = 1 + (i + 1) * lon_seg;
        for (int j = 0; j < lon_seg; ++j) {
            int j1 = (j + 1) % lon_seg;
            idx.push_back(row0 + j);  idx.push_back(row0 + j1); idx.push_back(row1 + j1);
            idx.push_back(row0 + j);  idx.push_back(row1 + j1); idx.push_back(row1 + j);
        }
    }
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
    sm.vbo.bind();
    sm.vao.attrib_pointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (const void*)0);
    sm.vao.enable_attrib(0);
    instance_buf.bind();
    sm.vao.attrib_pointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (const void*)0);
    sm.vao.enable_attrib(1);
    glVertexAttribDivisor(1, 1);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sm.ebo.handle());
    gl::VertexArray::unbind();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    gllib::log_to_stderr(gllib::LogLevel::info);
    gfx::Window window({"37 Point-Sampled Emissive Lighting", 1600, 900});
    window.vsync(false);

    gfx::ImGuiOverlay gui;
    if (!gui.init(window)) {
        gllib::log(gllib::LogLevel::error, "ImGui init failed");
        return EXIT_FAILURE;
    }

    // Hot-reloadable shaders (edit shaders/*.glsl -> auto recompile).
    gl::HotReloadProgram gbuf_prog;
    gbuf_prog.add_stage("shaders/gbuf.vert", gl::ShaderType::vertex);
    gbuf_prog.add_stage("shaders/gbuf.frag", gl::ShaderType::fragment);
    gl::HotReloadProgram display_prog;
    display_prog.add_stage("shaders/display.vert", gl::ShaderType::vertex);
    display_prog.add_stage("shaders/display.frag", gl::ShaderType::fragment);
    gl::HotReloadProgram pc_prog;
    pc_prog.add_stage("shaders/pc.vert", gl::ShaderType::vertex);
    pc_prog.add_stage("shaders/pc.frag", gl::ShaderType::fragment);
    gl::HotReloadProgram sphere_prog;
    sphere_prog.add_stage("shaders/sphere.vert", gl::ShaderType::vertex);
    sphere_prog.add_stage("shaders/sphere.frag", gl::ShaderType::fragment);
    gl::HotReloadProgram leaf_refit_prog;
    leaf_refit_prog.add_stage("shaders/leaf_refit.comp", gl::ShaderType::compute);
    gl::HotReloadProgram refit_up_prog;
    refit_up_prog.add_stage("shaders/refit_bottom_up.comp", gl::ShaderType::compute);
    gl::HotReloadProgram rad_seed_prog;
    rad_seed_prog.add_stage("shaders/radiance_seed.comp", gl::ShaderType::compute);
    gl::HotReloadProgram rad_pull_prog;
    rad_pull_prog.add_stage("shaders/radiance_pull.comp", gl::ShaderType::compute);
    gl::HotReloadProgram rad_bounce_prog;
    rad_bounce_prog.add_stage("shaders/radiance_bounce.comp", gl::ShaderType::compute);
    gl::HotReloadProgram micro_render_prog;
    micro_render_prog.add_stage("shaders/micro_render.comp", gl::ShaderType::compute);
    gl::HotReloadProgram bilateral_prog;
    bilateral_prog.add_stage("shaders/bilateral_upsample.comp", gl::ShaderType::compute);
    gbuf_prog.poll();
    display_prog.poll();
    pc_prog.poll();
    sphere_prog.poll();
    leaf_refit_prog.poll();
    refit_up_prog.poll();
    rad_seed_prog.poll();
    rad_pull_prog.poll();
    rad_bounce_prog.poll();
    micro_render_prog.poll();
    bilateral_prog.poll();
    auto gbuf = gbuf_prog.take_program();
    auto display = display_prog.take_program();
    auto pc = pc_prog.take_program();
    auto spherep = sphere_prog.take_program();
    auto leaf_refit = leaf_refit_prog.take_program();
    auto refit_up = refit_up_prog.take_program();
    auto rad_seed = rad_seed_prog.take_program();
    auto rad_pull = rad_pull_prog.take_program();
    auto rad_bounce = rad_bounce_prog.take_program();
    auto micro_render = micro_render_prog.take_program();
    auto bilateral = bilateral_prog.take_program();
    if (!gbuf || !display || !pc || !spherep ||
        !leaf_refit || !refit_up || !rad_seed || !rad_pull || !rad_bounce ||
        !micro_render || !bilateral)
        return EXIT_FAILURE;

    // Fullscreen triangle for the display pass (no vertex buffer).
    gl::VertexArray fsq_vao;

    // Camera.
    gfx::Camera cam;
    cam.perspective(45.0f, float(window.framebuffer_width()) / float(window.framebuffer_height()),
                    0.1f, 1000.0f);
    cam.look_at(glm::vec3(0, 1.5f, 4.5f), glm::vec3(0, 0.5f, 0));

    // Model state.
    int model_index = 0;
    gfx::Model model;
    glm::mat4 model_mat(1.0f);
    float cam_dist = 4.5f;

    auto load_model = [&](int index) {
        const char* path = kModels[index].path;
        gfx::Model fresh;
        if (!fresh.load(path)) {
            gllib::logf(gllib::LogLevel::error, "Failed to load %s", path);
            return;
        }
        model = std::move(fresh);
        model_mat = model_transform(model, kModels[index].path, 2.0f);
        gllib::logf(gllib::LogLevel::info, "Loaded %s: %zu meshes, %zu materials",
                    kModels[index].label, model.mesh_count(), model.material_count());
        cam_dist = 4.5f;
        cam.set_position(glm::vec3(0, 1.5f, cam_dist));
        cam.set_target(glm::vec3(0, 0.5f, 0));
    };
    load_model(model_index);

    // ---- S3: surfel octree buffers ----
    gl::Buffer pgeom_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer pnrm_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer palb_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer pemit_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer leaf_packed_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer sphere_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer cone_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer packed_bounds_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer packed_cone_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer packed_sphere_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    // T1b: merged 16 B/node record (sphere + cone + emissive flag) for the traversal.
    gl::Buffer node_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    // Step 0: striped visit statistics (64 uvec4 slots).
    gl::Buffer visit_stats_buf(gl::BufferType::shader, gl::BufferUsage::stream_read);
    {
        std::vector<uint32_t> zeros(64 * 4, 0u);
        visit_stats_buf.data(zeros.data(), zeros.size() * sizeof(uint32_t));
    }

    // ---- S4: refit + radiance buffers ----
    gl::Buffer tri_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);      // TriangleBuf
    gl::Buffer leaf_src_buf(gl::BufferType::shader, gl::BufferUsage::static_draw);  // LeafSource
    gl::Buffer radiance_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw); // per-node RGBA16F
    // Bounce re-emission target (leaf region), copied back into radiance_buf.
    gl::Buffer rad_next_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);

    gl::VertexArray pc_vao;   // empty VAO: point cloud reads SSBOs via gl_VertexID

    // S3 state (CPU octree lives at main scope so S4 can validate against it)
    int point_budget_index = 1;
    int N = 0;
    int oct_total_nodes = 0, oct_interior = 0;
    float leaf_radius = 0.0f;
    PackParams pp;
    bool have_hierarchy = false;
    Octree8 oct;                 // CPU reference (built/cached, used for validation)
    std::vector<Tri> tris;

    auto build_hierarchy = [&]() {
        N = kBudgetOptions[point_budget_index];
        const std::string& model_path = kModels[model_index].path;

        CacheKey key;
        key.model_stamp = file_stamp(model_path, nullptr);
        key.N = uint32_t(N);
        key.mat_hash = material_hash(model);
        key.transform_hash = matrix_hash(model_mat);

        std::string cpath = cache_path_for(model_path, N);

        if (load_cache(cpath, key, oct, tris)) {
            gllib::logf(gllib::LogLevel::info, "S3: cache hit %s (N=%d)", cpath.c_str(), N);
        } else {
            auto t0 = std::chrono::steady_clock::now();
            gllib::logf(gllib::LogLevel::info, "S3: building hierarchy N=%d ...", N);
            tris = extract_triangles(model, model_mat);
            auto t1 = std::chrono::steady_clock::now();
            float total_area = 0.0f;
            for (const auto& t : tris) total_area += t.area;
            auto pts = sample_points_mitchell(tris, total_area, N, kBEST_K);
            auto t2 = std::chrono::steady_clock::now();
            oct = build_octree8(std::move(pts), total_area, kCOVERAGE);
            auto t3 = std::chrono::steady_clock::now();
            if (save_cache(cpath, key, oct, tris))
                gllib::logf(gllib::LogLevel::info, "S3: cached to %s", cpath.c_str());
            auto t4 = std::chrono::steady_clock::now();
            auto ms = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            gllib::logf(gllib::LogLevel::info,
                        "S3: built in %.0f ms (extract=%.0f sample=%.0f tree=%.0f cache=%.0f) "
                        "(%zu tris, N=%d, leaf_radius=%.5f)",
                        ms(t0, t4), ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4),
                        tris.size(), N, oct.leaf_radius);
        }

        oct_total_nodes = oct.total_nodes;
        oct_interior = oct.interior;
        leaf_radius = oct.leaf_radius;

        // PackParams from leaf bounds
        glm::vec3 mn(1e30f), mx(-1e30f);
        for (const auto& s : oct.pts) {
            mn = glm::min(mn, s.pos);
            mx = glm::max(mx, s.pos);
        }
        pp.scene_min = mn;
        pp.scene_size = mx - mn;
        pp.radius_scale = glm::length(mx - mn);

        // Sanity: surfel bounds should match the transformed mesh vertex bounds.
        glm::vec3 mlo(1e30f), mhi(-1e30f);
        for (size_t i = 0; i < model.mesh_count(); ++i)
            for (const auto& v : model.mesh(int(i)).vertices())
                for (int c = 0; c < 3; ++c) {
                    float w = glm::vec3(model_mat * glm::vec4(v.position[0], v.position[1], v.position[2], 1.0f))[c];
                    mlo[c] = std::min(mlo[c], w);
                    mhi[c] = std::max(mhi[c], w);
                }
        gllib::logf(gllib::LogLevel::info,
                    "S3 sanity: surfel AABB (%.4f %.4f %.4f)-(%.4f %.4f %.4f) | mesh AABB (%.4f %.4f %.4f)-(%.4f %.4f %.4f)",
                    mn.x, mn.y, mn.z, mx.x, mx.y, mx.z, mlo.x, mlo.y, mlo.z, mhi.x, mhi.y, mhi.z);

        // fp32 leaf buffers
        std::vector<glm::vec4> vp(N), vn(N), va(N), ve(N);
        for (int i = 0; i < N; ++i) {
            vp[i] = glm::vec4(oct.pts[i].pos, oct.leaf_radius);
            vn[i] = glm::vec4(oct.pts[i].nrm, 0.0f);
            va[i] = glm::vec4(oct.pts[i].alb, 1.0f);
            ve[i] = glm::vec4(oct.pts[i].emi, 1.0f);
        }
        pgeom_buf.data(vp.data(), vp.size() * sizeof(glm::vec4));
        pnrm_buf.data(vn.data(), vn.size() * sizeof(glm::vec4));
        palb_buf.data(va.data(), va.size() * sizeof(glm::vec4));
        pemit_buf.data(ve.data(), ve.size() * sizeof(glm::vec4));
        sphere_buf.data(oct.sphere.data(), oct.sphere.size() * sizeof(glm::vec4));
        cone_buf.data(oct.cone.data(), oct.cone.size() * sizeof(glm::vec4));

        // Packed leaf records + validation vs fp32
        std::vector<PackedLeaf> pleaf(N);
        double max_pos_err = 0.0, max_nrm_err = 0.0, max_alb_err = 0.0, max_r_err = 0.0;
        for (int i = 0; i < N; ++i) {
            pack_leaf(oct.pts[i], oct.leaf_radius, pp, pleaf[i]);
            // decode & compare
            PackedLeaf& L = pleaf[i];
            glm::vec3 dpos = pp.scene_min + pp.scene_size *
                (glm::vec3(float(L.xy >> 16), float(L.zw >> 16), float(L.zw & 0xFFFFu)) / 65535.0f);
            float dr = float(L.xy & 0xFFFFu) / 65535.0f * pp.radius_scale;
            int hi = int(L.nrm >> 16); hi = hi > 32767 ? hi - 65536 : hi;
            int lo = int(L.nrm & 0xFFFFu); lo = lo > 32767 ? lo - 65536 : lo;
            glm::vec3 o2((float(hi) / 32767.0f), (float(lo) / 32767.0f), 0.0f);
            o2.z = 1.0f - std::abs(o2.x) - std::abs(o2.y);
            if (o2.z < 0.0f) { float tx = o2.x, ty = o2.y; o2.x = (1.0f - std::abs(ty)) * (tx >= 0 ? 1 : -1); o2.y = (1.0f - std::abs(tx)) * (ty >= 0 ? 1 : -1); }
            o2 = glm::normalize(o2);
            glm::vec3 dalb{ float((L.alb >> 16) & 0xFFu) / 255.0f,
                            float((L.alb >> 8) & 0xFFu) / 255.0f,
                            float(L.alb & 0xFFu) / 255.0f };
            max_pos_err = std::max(max_pos_err, double(glm::length(dpos - oct.pts[i].pos)));
            max_nrm_err = std::max(max_nrm_err, double(glm::length(o2 - oct.pts[i].nrm)));
            max_alb_err = std::max(max_alb_err, double(glm::length(dalb - oct.pts[i].alb)));
            max_r_err = std::max(max_r_err, double(std::abs(dr - oct.leaf_radius)));
        }
        leaf_packed_buf.data(pleaf.data(), pleaf.size() * sizeof(PackedLeaf));
        gllib::logf(gllib::LogLevel::info,
                    "S3b packed validation: max err pos=%.6f nrm=%.4f alb=%.3f radius=%.6f",
                    max_pos_err, max_nrm_err, max_alb_err, max_r_err);

        // Packed interior bounds + cones (uploaded for later stages; validation
        // of leaf packing above is the S3b gate).
        std::vector<PackedBounds> pbounds(oct.interior);
        for (int node = 0; node < oct.interior; ++node) {
            glm::vec3 centers[8]; float radii[8];
            for (int c = 0; c < 8; ++c) {
                int ch = 8 * node + 1 + c;
                centers[c] = glm::vec3(oct.sphere[ch]);
                radii[c] = oct.sphere[ch].w;
            }
            pack_bounds(centers, radii, pp, pbounds[node]);
        }
        packed_bounds_buf.data(pbounds.data(), pbounds.size() * sizeof(PackedBounds));

        std::vector<PackedCone> pcone(oct.total_nodes);
        for (int node = 0; node < oct.total_nodes; ++node) {
            pack_cone(glm::vec3(oct.cone[node]), oct.cone[node].w, pcone[node]);
        }
        packed_cone_buf.data(pcone.data(), pcone.size() * sizeof(PackedCone));

        // Per-node packed spheres (8 B/node) for the traversal (P3).
        std::vector<PackedSphere> psphere(oct.total_nodes);
        for (int node = 0; node < oct.total_nodes; ++node) {
            glm::vec3 t = glm::clamp((glm::vec3(oct.sphere[node]) - pp.scene_min) / pp.scene_size, 0.0f, 1.0f);
            uint16_t px = q_u16(t.x), py = q_u16(t.y), pz = q_u16(t.z);
            uint16_t r = q_u16(std::clamp(oct.sphere[node].w / pp.radius_scale, 0.0f, 1.0f));
            psphere[node].x = (uint32_t(px) << 16) | uint32_t(r);
            psphere[node].y = (uint32_t(py) << 16) | uint32_t(pz);
        }
        packed_sphere_buf.data(psphere.data(), psphere.size() * sizeof(PackedSphere));

        // P3 validation: decode packed sphere/cone and compare to fp32.
        {
            double max_sph = 0.0, max_axis = 0.0, max_cw = 0.0;
            for (int node = 0; node < oct.total_nodes; ++node) {
                glm::vec3 dpos = pp.scene_min + pp.scene_size *
                    (glm::vec3(float(psphere[node].x >> 16), float(psphere[node].y >> 16),
                               float(psphere[node].y & 0xFFFFu)) / 65535.0f);
                float dr = float(psphere[node].x & 0xFFFFu) / 65535.0f * pp.radius_scale;
                max_sph = std::max(max_sph, double(glm::length(dpos - glm::vec3(oct.sphere[node]))));
                max_sph = std::max(max_sph, double(std::abs(dr - oct.sphere[node].w)));

                int hi = int(pcone[node].axis[0]); hi = hi > 32767 ? hi - 65536 : hi;
                int lo = int(pcone[node].axis[1]); lo = lo > 32767 ? lo - 65536 : lo;
                glm::vec3 o2(float(hi) / 32767.0f, float(lo) / 32767.0f, 0.0f);
                o2.z = 1.0f - std::abs(o2.x) - std::abs(o2.y);
                if (o2.z < 0.0f) { float tx = o2.x, ty = o2.y; o2.x = (1.0f - std::abs(ty)) * (tx >= 0 ? 1 : -1); o2.y = (1.0f - std::abs(tx)) * (ty >= 0 ? 1 : -1); }
                o2 = glm::normalize(o2);
                max_axis = std::max(max_axis, double(glm::length(o2 - glm::vec3(oct.cone[node]))));
                float dcw = float(pcone[node].half) / 255.0f * 2.0f - 1.0f;
                max_cw = std::max(max_cw, double(std::abs(dcw - oct.cone[node].w)));
            }
            gllib::logf(gllib::LogLevel::info, "P3 packed validate: sphere err %.6f | axis err %.5f | cw err %.5f",
                        max_sph, max_axis, max_cw);
        }

        // ---- T1b/T2b: merged node records with emissive subtree flags ----
        // .xy = packed sphere, .z = cone axis pair, .w = cone half | hasEmissive<<16
        std::vector<uint32_t> eflag(oct.total_nodes, 0u);
        for (int i = 0; i < N; ++i)
            eflag[oct.interior + i] =
                (glm::dot(oct.pts[i].emi, oct.pts[i].emi) > 1e-8f) ? 1u : 0u;
        for (int node = oct.interior - 1; node >= 0; --node) {
            uint32_t f = 0u;
            for (int c = 0; c < 8; ++c) f |= eflag[8 * node + 1 + c];
            eflag[node] = f;
        }
        size_t emis_count = 0;
        for (int node = 0; node < oct.total_nodes; ++node)
            if (eflag[node]) emis_count++;

        std::vector<glm::uvec4> node_rec(oct.total_nodes);
        for (int node = 0; node < oct.total_nodes; ++node) {
            node_rec[node] = glm::uvec4(
                psphere[node].x, psphere[node].y,
                (uint32_t(pcone[node].axis[0]) << 16) | uint32_t(pcone[node].axis[1]),
                uint32_t(pcone[node].half) | (eflag[node] << 16));
        }
        node_buf.data(node_rec.data(), node_rec.size() * sizeof(glm::uvec4));
        gllib::logf(gllib::LogLevel::info,
                    "T2b: %zu / %d nodes carry emissive (%.2f%%)",
                    emis_count, oct.total_nodes,
                    100.0 * double(emis_count) / double(oct.total_nodes));

        // Bind for shaders
        pgeom_buf.bind_base(0);
        pnrm_buf.bind_base(1);
        palb_buf.bind_base(2);
        pemit_buf.bind_base(3);
        leaf_packed_buf.bind_base(4);
        sphere_buf.bind_base(5);
        tri_buf.bind_base(6);
        leaf_src_buf.bind_base(7);
        radiance_buf.bind_base(8);
        cone_buf.bind_base(9);
        node_buf.bind_base(10);
        visit_stats_buf.bind_base(12);

        // ---- S4: upload TriangleBuf + LeafSource + seed Radiance ----
        std::vector<GpuTri> gpu_tris(tris.size());
        for (size_t i = 0; i < tris.size(); ++i) {
            for (int v = 0; v < 3; ++v) {
                gpu_tris[i].pos[v] = glm::vec4(tris[i].p[v], 0.0f);
                gpu_tris[i].nrm[v] = glm::vec4(tris[i].n[v], 0.0f);
            }
        }
        tri_buf.data(gpu_tris.data(), gpu_tris.size() * sizeof(GpuTri));

        std::vector<LeafSrc> leaf_src(N);
        for (int i = 0; i < N; ++i) {
            leaf_src[i] = { uint32_t(oct.pts[i].tri_idx), oct.pts[i].u, oct.pts[i].v, 0.0f };
        }
        leaf_src_buf.data(leaf_src.data(), leaf_src.size() * sizeof(LeafSrc));

        std::vector<glm::vec4> rad(oct.total_nodes, glm::vec4(0.0f));
        for (int i = 0; i < N; ++i) {
            int node = oct.interior + i;
            rad[node] = glm::vec4(oct.pts[i].emi, 1.0f);
        }
        radiance_buf.data(rad.data(), rad.size() * sizeof(glm::vec4));
        rad_next_buf.data(rad.data(), rad.size() * sizeof(glm::vec4));
        rad_next_buf.bind_base(13);

        have_hierarchy = true;
    };

    build_hierarchy();

    // Bounding-sphere wireframe overlay (reads sphere_buf via instance attrib).
    SphereMesh sphere_mesh = create_wireframe_sphere(8, 16);
    setup_sphere_vao(sphere_mesh, sphere_buf);

    // G-buffer + display state.
    FrameTargets targets;
    create_targets(targets, window.framebuffer_width(), window.framebuffer_height());

    // S5: micro-render gather output (full-res indirect).
    int gather_scale = 1;   // 1/2/4/16 -> gather resolution divisor
    // NOTE: image_2d allocates immutable storage; re-creating a texture at a new
    // size must go through a fresh object (move-assign deletes the old storage).
    gl::Texture indirect_tex{gl::TextureType::tex_2d};
    auto create_indirect_tex = [&](int w, int h) {
        int rw = std::max(1, w / gather_scale);
        int rh = std::max(1, h / gather_scale);
        indirect_tex = gl::Texture{gl::TextureType::tex_2d};
        indirect_tex.image_2d(0, GL_RGBA16F, rw, rh, GL_RGBA, GL_FLOAT, nullptr, 1);
        indirect_tex.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        indirect_tex.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        indirect_tex.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        indirect_tex.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };
    create_indirect_tex(window.framebuffer_width(), window.framebuffer_height());

    // P2b: full-res bilaterally-upsampled indirect output.
    gl::Texture indirect_upsampled_tex{gl::TextureType::tex_2d};
    auto create_upsampled_tex = [&](int w, int h) {
        indirect_upsampled_tex = gl::Texture{gl::TextureType::tex_2d};
        indirect_upsampled_tex.image_2d(0, GL_RGBA16F, w, h, GL_RGBA, GL_FLOAT, nullptr, 1);
        indirect_upsampled_tex.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        indirect_upsampled_tex.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        indirect_upsampled_tex.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        indirect_upsampled_tex.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };
    create_upsampled_tex(window.framebuffer_width(), window.framebuffer_height());

    int view_mode = 0;  // 0=position 1=normal 2=albedo 3=emissive 4=depth
    float exposure = 1.0f;
    float gamma = 2.2f;
    const float far_plane = 100.0f;
    bool captured = false;

    // S3 debug state
    bool show_points = true;
    float point_size = 4.0f;
    int pc_color_mode = 0;
    bool use_packed = false;
    bool show_spheres = false;
    int sphere_lod = 0;  // 0=all 1=interior 2=leaves
    glm::vec4 sphere_color(0.0f, 1.0f, 0.0f, 0.4f);

    // S4 state
    bool run_refit = true;
    bool run_radiance = true;
    float emissive_gain = 1.0f;
    int num_bounces = 0;  // multi-bounce wiring lands with S5; seeds only for now

    // S5 state
    bool run_micro_render = true;
    int micro_size = 8;
    float micro_gain = 1.0f;
    float bilateral_depth_sigma = 0.05f;
    float bilateral_normal_exp = 16.0f;

    // Micro-buffer atlas: one micro_size^2 tile of packed (depth<<19|node) per
    // gather point, depth-sorted with imageAtomicMin (replaces the DFS traversal).
    gl::Texture micro_atlas{gl::TextureType::tex_2d};
    auto create_micro_atlas = [&](int w, int h) {
        int rw = std::max(1, w / gather_scale);
        int rh = std::max(1, h / gather_scale);
        micro_atlas = gl::Texture{gl::TextureType::tex_2d};
        micro_atlas.image_2d(0, GL_R32UI, rw * micro_size, rh * micro_size,
                             GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr, 1);
        micro_atlas.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        micro_atlas.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        micro_atlas.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        micro_atlas.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };
    create_micro_atlas(window.framebuffer_width(), window.framebuffer_height());

    // Bounce micro-buffer atlas: one tile per LEAF gather point (depends on N and
    // micro_size; recreated lazily in the frame loop when either changes).
    gl::Texture bounce_atlas{gl::TextureType::tex_2d};
    int bounce_tiles_x = 1;
    auto create_bounce_atlas = [&]() {
        if (N <= 0) return;
        bounce_tiles_x = int(std::ceil(std::sqrt(float(N))));
        int tiles_y = (N + bounce_tiles_x - 1) / bounce_tiles_x;
        bounce_atlas = gl::Texture{gl::TextureType::tex_2d};
        bounce_atlas.image_2d(0, GL_R32UI, bounce_tiles_x * micro_size, tiles_y * micro_size,
                              GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr, 1);
        bounce_atlas.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        bounce_atlas.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        bounce_atlas.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        bounce_atlas.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };

    // Disk-pixel count for the current micro_size (shared by gather + bounce).
    auto m_valid_count = [&]() {
        int mv = 0;
        for (int ly = 0; ly < micro_size; ++ly)
            for (int lx = 0; lx < micro_size; ++lx) {
                float mu = (2.0f * lx + 1.0f) / micro_size - 1.0f;
                float mv2 = (2.0f * ly + 1.0f) / micro_size - 1.0f;
                if (mu * mu + mv2 * mv2 <= 1.0f) mv++;
            }
        return mv;
    };

    // P0: per-pass GPU timers
    GpuTimer t_gbuf("g-buffer");
    GpuTimer t_refit("refit");
    GpuTimer t_radiance("radiance");
    GpuTimer t_bounce("bounce");
    GpuTimer t_micro("micro-render");
    GpuTimer t_upsample("bilateral");
    GpuTimer t_display("display");

    // Read back GPU octree + radiance and compare against the CPU reference.
    auto validate_s4 = [&]() {
        if (!have_hierarchy || N == 0) return;
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();

        std::vector<glm::vec4> gpu_pgeom(N), gpu_sphere(oct_total_nodes), gpu_rad(oct_total_nodes);

        void* ptr = pgeom_buf.map_range(0, N * sizeof(glm::vec4), GL_MAP_READ_BIT);
        if (ptr) { std::memcpy(gpu_pgeom.data(), ptr, N * sizeof(glm::vec4)); pgeom_buf.unmap(); }
        ptr = sphere_buf.map_range(0, oct_total_nodes * sizeof(glm::vec4), GL_MAP_READ_BIT);
        if (ptr) { std::memcpy(gpu_sphere.data(), ptr, oct_total_nodes * sizeof(glm::vec4)); sphere_buf.unmap(); }
        ptr = radiance_buf.map_range(0, oct_total_nodes * sizeof(glm::vec4), GL_MAP_READ_BIT);
        if (ptr) { std::memcpy(gpu_rad.data(), ptr, oct_total_nodes * sizeof(glm::vec4)); radiance_buf.unmap(); }

        // Expected radiance: emissive leaves x gain, pull-up average.
        std::vector<glm::vec3> exp_rad(oct_total_nodes, glm::vec3(0.0f));
        for (int i = 0; i < N; ++i)
            exp_rad[oct_interior + i] = oct.pts[i].emi * emissive_gain;
        for (int node = oct_interior - 1; node >= 0; --node) {
            for (int c = 0; c < 8; ++c)
                exp_rad[node] += exp_rad[8 * node + 1 + c];
            exp_rad[node] /= 8.0f;
        }

        double max_pos = 0.0, max_sph = 0.0, max_rad = 0.0;
        for (int i = 0; i < N; ++i)
            max_pos = std::max(max_pos, double(glm::length(glm::vec3(gpu_pgeom[i]) - oct.pts[i].pos)));
        for (int node = 0; node < oct_total_nodes; ++node) {
            max_sph = std::max(max_sph, double(glm::length(gpu_sphere[node] - oct.sphere[node])));
            max_rad = std::max(max_rad, double(glm::length(glm::vec3(gpu_rad[node]) - exp_rad[node])));
        }
        gllib::logf(gllib::LogLevel::info,
                    "S4 validate (N=%d): leaf pos err %.6f | sphere err %.6f | radiance err %.6f (gain=%.2f)",
                    N, max_pos, max_sph, max_rad, emissive_gain);
    };

    // Read back the S5 gather output and report mean/max indirect radiance.
    auto validate_s5 = [&]() {
        if (!have_hierarchy) return;
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();
        int rw = std::max(1, targets.w / gather_scale);
        int rh = std::max(1, targets.h / gather_scale);
        std::vector<glm::vec4> px(size_t(rw) * size_t(rh));
        glBindTexture(GL_TEXTURE_2D, indirect_tex.handle());
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, px.data());
        glBindTexture(GL_TEXTURE_2D, 0);
        double mean = 0.0, maxv = 0.0;
        for (const auto& v : px) {
            double m = double(v.r + v.g + v.b) / 3.0;
            mean += m;
            maxv = std::max(maxv, m);
        }
        mean /= double(rw) * double(rh);
        gllib::logf(gllib::LogLevel::info,
                    "S5 indirect stats (N=%d, micro=%d, gain=%.2f, scale=%d): mean=%.5f max=%.5f",
                    N, micro_size, micro_gain, gather_scale, mean, maxv);
    };

    double last = window.time();
    uint64_t frame_counter = 0;

    while (!window.should_close()) {
        double now = window.time();
        float dt = float(now - last);
        last = now;
        ++frame_counter;

        window.poll_events();
        camera_control(window, cam, dt, !gui.wants_mouse(), captured);
        cam.set_aspect(float(window.framebuffer_width()) / float(window.framebuffer_height()));

        int fw = window.framebuffer_width(), fh = window.framebuffer_height();
        static int prev_gather_scale = gather_scale;
        static int prev_micro_size = micro_size;
        static int prev_bounce_N = -1;
        if (fw != targets.w || fh != targets.h) {
            create_targets(targets, fw, fh);
            create_indirect_tex(fw, fh);
            create_upsampled_tex(fw, fh);
            create_micro_atlas(fw, fh);
        } else if (gather_scale != prev_gather_scale || micro_size != prev_micro_size) {
            create_indirect_tex(fw, fh);
            create_micro_atlas(fw, fh);
        }
        if (N != prev_bounce_N || micro_size != prev_micro_size) create_bounce_atlas();
        prev_gather_scale = gather_scale;
        prev_micro_size = micro_size;
        prev_bounce_N = N;

        if (gbuf_prog.poll()) gbuf = gbuf_prog.take_program();
        if (display_prog.poll()) display = display_prog.take_program();
        if (pc_prog.poll()) pc = pc_prog.take_program();
        if (sphere_prog.poll()) spherep = sphere_prog.take_program();
        if (leaf_refit_prog.poll()) leaf_refit = leaf_refit_prog.take_program();
        if (refit_up_prog.poll()) refit_up = refit_up_prog.take_program();
        if (rad_seed_prog.poll()) rad_seed = rad_seed_prog.take_program();
        if (rad_pull_prog.poll()) rad_pull = rad_pull_prog.take_program();
        if (rad_bounce_prog.poll()) rad_bounce = rad_bounce_prog.take_program();
        if (micro_render_prog.poll()) micro_render = micro_render_prog.take_program();
        if (bilateral_prog.poll()) bilateral = bilateral_prog.take_program();

        glm::mat4 vp = cam.view_projection();

        // ---- 1. Forward G-buffer pass ----
        t_gbuf.begin();
        targets.fbo.bind();
        gl::viewport(0, 0, targets.w, targets.h);
        gl::clear_color(0.0f, 0.0f, 0.0f, 0.0f);
        gl::clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        gl::enable(GL_DEPTH_TEST);
        gl::depth_func(GL_LESS);

        if (gbuf->valid()) {
            gbuf->use();
            auto gloc = [&](const char* n) { return gbuf->uniform_location(n); };
            GLint loc;
            loc = gloc("u_view_proj"); if (loc >= 0) gbuf->uniform_matrix4fv(loc, glm::value_ptr(vp));
            loc = gloc("u_view");      if (loc >= 0) gbuf->uniform_matrix4fv(loc, glm::value_ptr(cam.view()));
            loc = gloc("u_model");     if (loc >= 0) gbuf->uniform_matrix4fv(loc, glm::value_ptr(model_mat));
            glm::mat3 normal_mat = glm::transpose(glm::inverse(glm::mat3(model_mat)));
            loc = gloc("u_normal_mat"); if (loc >= 0) gbuf->uniform_matrix3fv(loc, glm::value_ptr(normal_mat));

            for (size_t i = 0; i < model.mesh_count(); ++i) {
                int mi = model.mesh_material(int(i));
                const auto& mat = model.material_info(size_t(mi >= 0 ? mi : 0));

                loc = gloc("u_albedo");   if (loc >= 0) gbuf->uniform3fv(loc, mat.base_color_factor);
                loc = gloc("u_emissive"); if (loc >= 0) gbuf->uniform3fv(loc, mat.emissive_factor);

                if (mat.base_color_tex >= 0 && size_t(mat.base_color_tex) < model.texture_count()) {
                    model.texture(mat.base_color_tex)->bind(0);
                    loc = gloc("u_base_tex");     if (loc >= 0) gbuf->uniform1i(loc, 0);
                    loc = gloc("u_has_base_tex"); if (loc >= 0) gbuf->uniform1i(loc, 1);
                } else {
                    loc = gloc("u_has_base_tex"); if (loc >= 0) gbuf->uniform1i(loc, 0);
                }

                if (mat.emissive_tex >= 0 && size_t(mat.emissive_tex) < model.texture_count()) {
                    model.texture(mat.emissive_tex)->bind(1);
                    loc = gloc("u_emissive_tex");     if (loc >= 0) gbuf->uniform1i(loc, 1);
                    loc = gloc("u_has_emissive_tex"); if (loc >= 0) gbuf->uniform1i(loc, 1);
                } else {
                    loc = gloc("u_has_emissive_tex"); if (loc >= 0) gbuf->uniform1i(loc, 0);
                }

                model.mesh(i).draw();
            }
        }
        gl::Framebuffer::unbind(gl::FramebufferType::both);
        t_gbuf.end();

        // ---- 1b. S4: GPU refit + emissive radiance seed/pull-up ----
        if (have_hierarchy && (run_refit || run_radiance)) {
            if (run_refit && leaf_refit->valid() && refit_up->valid()) {
                t_refit.begin();
                leaf_refit->use();
                auto ul = [&](const char* n) { return leaf_refit->uniform_location(n); };
                GLint loc;
                loc = ul("u_num_leaves");   if (loc >= 0) leaf_refit->uniform1ui(loc, GLuint(N));
                loc = ul("u_tree_offset");  if (loc >= 0) leaf_refit->uniform1ui(loc, GLuint(oct_interior));
                loc = ul("u_leaf_radius");  if (loc >= 0) leaf_refit->uniform1f(loc, leaf_radius);
                gl::dispatch_compute((N + 255) / 256, 1, 1);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                for (int d = oct.depth - 1; d >= 0; --d) {
                    uint32_t count = 1u << (3 * d);
                    uint32_t level_start = (count - 1) / 7;
                    refit_up->use();
                    auto rl = [&](const char* n) { return refit_up->uniform_location(n); };
                    loc = rl("u_count");        if (loc >= 0) refit_up->uniform1ui(loc, count);
                    loc = rl("u_level_start");  if (loc >= 0) refit_up->uniform1ui(loc, level_start);
                    gl::dispatch_compute((count + 255) / 256, 1, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                }
                t_refit.end();
            }

            if (run_radiance && rad_seed->valid() && rad_pull->valid()) {
                t_radiance.begin();
                rad_seed->use();
                auto sl = [&](const char* n) { return rad_seed->uniform_location(n); };
                GLint loc;
                loc = sl("u_num_leaves");   if (loc >= 0) rad_seed->uniform1ui(loc, GLuint(N));
                loc = sl("u_tree_offset");  if (loc >= 0) rad_seed->uniform1ui(loc, GLuint(oct_interior));
                loc = sl("u_gain");         if (loc >= 0) rad_seed->uniform1f(loc, emissive_gain);
                gl::dispatch_compute((N + 255) / 256, 1, 1);
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                for (int d = oct.depth - 1; d >= 0; --d) {
                    uint32_t count = 1u << (3 * d);
                    uint32_t level_start = (count - 1) / 7;
                    rad_pull->use();
                    auto pl = [&](const char* n) { return rad_pull->uniform_location(n); };
                    loc = pl("u_count");        if (loc >= 0) rad_pull->uniform1ui(loc, count);
                    loc = pl("u_level_start");  if (loc >= 0) rad_pull->uniform1ui(loc, level_start);
                    gl::dispatch_compute((count + 255) / 256, 1, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                }
                t_radiance.end();

                // ---- Re-emission bounces: rad_next = emi + albedo * gather(rad) ----
                if (num_bounces > 0 && rad_bounce->valid()) {
                    static bool warned_large = false;
                    if (!warned_large && N > 32768) {
                        warned_large = true;
                        gllib::logf(gllib::LogLevel::warn,
                                    "Bounce gather is O(N^2): N=%d will be very slow", N);
                    }
                    t_bounce.begin();
                    int m_valid = m_valid_count();
                    for (int b = 0; b < num_bounces; ++b) {
                        uint32_t empty = 0xFFFFFFFFu;
                        glClearTexImage(bounce_atlas.handle(), 0, GL_RED_INTEGER,
                                        GL_UNSIGNED_INT, &empty);
                        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
                        bounce_atlas.bind_image(2, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);

                        rad_bounce->use();
                        auto bl = [&](const char* n) { return rad_bounce->uniform_location(n); };
                        GLint bloc;
                        bloc = bl("u_num_leaves");   if (bloc >= 0) rad_bounce->uniform1ui(bloc, GLuint(N));
                        bloc = bl("u_rad_offset");   if (bloc >= 0) rad_bounce->uniform1ui(bloc, GLuint(oct_interior));
                        bloc = bl("u_micro_size");   if (bloc >= 0) rad_bounce->uniform1ui(bloc, GLuint(micro_size));
                        bloc = bl("u_m_valid");      if (bloc >= 0) rad_bounce->uniform1ui(bloc, GLuint(m_valid));
                        bloc = bl("u_tiles_x");      if (bloc >= 0) rad_bounce->uniform1ui(bloc, GLuint(bounce_tiles_x));
                        bloc = bl("u_gain");         if (bloc >= 0) rad_bounce->uniform1f(bloc, emissive_gain);
                        bloc = bl("u_scene_min");    if (bloc >= 0) rad_bounce->uniform3f(bloc, pp.scene_min.x, pp.scene_min.y, pp.scene_min.z);
                        bloc = bl("u_scene_size");   if (bloc >= 0) rad_bounce->uniform3f(bloc, pp.scene_size.x, pp.scene_size.y, pp.scene_size.z);
                        bloc = bl("u_radius_scale"); if (bloc >= 0) rad_bounce->uniform1f(bloc, pp.radius_scale);
                        gl::dispatch_compute((N + 255) / 256, 1, 1);
                        glMemoryBarrier(GL_ALL_BARRIER_BITS);

                        // Copy the re-emitted leaf region back into Radiance.
                        glCopyNamedBufferSubData(rad_next_buf.handle(), radiance_buf.handle(),
                                                 GLintptr(oct_interior) * sizeof(glm::vec4),
                                                 GLintptr(oct_interior) * sizeof(glm::vec4),
                                                 GLsizeiptr(N) * sizeof(glm::vec4));
                        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    }

                    // Refresh the interior pull-up so downstream views stay consistent.
                    for (int d = oct.depth - 1; d >= 0; --d) {
                        uint32_t count = 1u << (3 * d);
                        uint32_t level_start = (count - 1) / 7;
                        rad_pull->use();
                        auto pl = [&](const char* n) { return rad_pull->uniform_location(n); };
                        GLint loc2;
                        loc2 = pl("u_count");        if (loc2 >= 0) rad_pull->uniform1ui(loc2, count);
                        loc2 = pl("u_level_start");  if (loc2 >= 0) rad_pull->uniform1ui(loc2, level_start);
                        gl::dispatch_compute((count + 255) / 256, 1, 1);
                        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    }
                    t_bounce.end();
                }
            }
        }

        // ---- 1c. S5: micro-render final gathering ----
        if (have_hierarchy && run_micro_render && micro_render->valid()) {
            t_micro.begin();
            int rw = std::max(1, fw / gather_scale);
            int rh = std::max(1, fh / gather_scale);
            int m_valid = m_valid_count();

            // Clear the micro-buffer atlas to "empty" (0xFFFFFFFF) before splatting.
            uint32_t empty = 0xFFFFFFFFu;
            glClearTexImage(micro_atlas.handle(), 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &empty);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

            indirect_tex.bind_image(0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
            micro_atlas.bind_image(1, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
            targets.position.bind(0);
            targets.normal.bind(1);
            targets.albedo.bind(2);

            micro_render->use();
            auto ml = [&](const char* n) { return micro_render->uniform_location(n); };
            GLint loc;
            GLint gsize[2] = {rw, rh};
            GLint fsize[2] = {fw, fh};
            loc = ml("u_screen_size");   if (loc >= 0) micro_render->uniform2iv(loc, gsize);
            loc = ml("u_full_size");     if (loc >= 0) micro_render->uniform2iv(loc, fsize);
            loc = ml("u_micro_size");    if (loc >= 0) micro_render->uniform1ui(loc, GLuint(micro_size));
            loc = ml("u_m_valid");       if (loc >= 0) micro_render->uniform1ui(loc, GLuint(m_valid));
            loc = ml("u_scale");         if (loc >= 0) micro_render->uniform1ui(loc, GLuint(gather_scale));
            loc = ml("u_gain");          if (loc >= 0) micro_render->uniform1f(loc, micro_gain);
            loc = ml("u_jitter_seed");   if (loc >= 0) micro_render->uniform1ui(loc, GLuint(frame_counter & 0xFFFFu));
            loc = ml("u_scene_min");     if (loc >= 0) micro_render->uniform3f(loc, pp.scene_min.x, pp.scene_min.y, pp.scene_min.z);
            loc = ml("u_scene_size");    if (loc >= 0) micro_render->uniform3f(loc, pp.scene_size.x, pp.scene_size.y, pp.scene_size.z);
            loc = ml("u_radius_scale");  if (loc >= 0) micro_render->uniform1f(loc, pp.radius_scale);
            loc = ml("u_num_leaves");    if (loc >= 0) micro_render->uniform1ui(loc, GLuint(N));
            loc = ml("u_rad_offset");    if (loc >= 0) micro_render->uniform1ui(loc, GLuint(oct_interior));

            gl::dispatch_compute((rw + 7) / 8, (rh + 7) / 8, 1);
            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            t_micro.end();
        }

        // ---- 1d. Bilateral upsampling (P2b) ----
        if (have_hierarchy && run_micro_render && gather_scale > 1 && bilateral->valid()) {
            t_upsample.begin();
            int rw = std::max(1, fw / gather_scale);
            int rh = std::max(1, fh / gather_scale);

            indirect_tex.bind(0);
            targets.position.bind(1);
            targets.normal.bind(2);
            indirect_upsampled_tex.bind_image(0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

            bilateral->use();
            auto bl = [&](const char* n) { return bilateral->uniform_location(n); };
            GLint loc;
            GLint lsize[2] = {rw, rh};
            GLint hsize[2] = {fw, fh};
            loc = bl("u_low_size");     if (loc >= 0) bilateral->uniform2iv(loc, lsize);
            loc = bl("u_hi_size");      if (loc >= 0) bilateral->uniform2iv(loc, hsize);
            loc = bl("u_depth_sigma");  if (loc >= 0) bilateral->uniform1f(loc, bilateral_depth_sigma);
            loc = bl("u_normal_exp");   if (loc >= 0) bilateral->uniform1f(loc, bilateral_normal_exp);

            gl::dispatch_compute((fw + 7) / 8, (fh + 7) / 8, 1);
            glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            t_upsample.end();
        }

        // ---- 2. Display pass ----
        t_display.begin();
        gl::disable(GL_DEPTH_TEST);
        gl::viewport(0, 0, fw, fh);
        gl::clear_color(0.02f, 0.02f, 0.03f, 1.0f);
        gl::clear(GL_COLOR_BUFFER_BIT);

        gl::Program* display_ptr = display->valid() ? display.get() : nullptr;
        if (display_ptr) {
            display_ptr->use();
            auto dloc = [&](const char* n) { return display_ptr->uniform_location(n); };
            GLint loc;
            gl::Texture* targets_list[] = {&targets.position, &targets.normal, &targets.albedo,
                                           &targets.emissive, &targets.depth, &indirect_tex};
            if (view_mode == 5 && gather_scale > 1)
                indirect_upsampled_tex.bind(0);
            else
                targets_list[view_mode]->bind(0);
            loc = dloc("u_tex");      if (loc >= 0) display_ptr->uniform1i(loc, 0);
            loc = dloc("u_mode");     if (loc >= 0) display_ptr->uniform1i(loc, view_mode);
            loc = dloc("u_far");      if (loc >= 0) display_ptr->uniform1f(loc, far_plane);
            loc = dloc("u_exposure"); if (loc >= 0) display_ptr->uniform1f(loc, exposure);
            loc = dloc("u_gamma");    if (loc >= 0) display_ptr->uniform1f(loc, gamma);
            fsq_vao.bind();
            gl::draw_arrays(GL_TRIANGLES, 0, 3);
        }
        t_display.end();

        // ---- 2b. Point cloud overlay (S3) ----
        if (show_points && have_hierarchy) {
            gl::Program* pcp = pc->valid() ? pc.get() : nullptr;
            if (pcp) {
                pcp->use();
                auto ploc = [&](const char* n) { return pcp->uniform_location(n); };
                GLint loc;
                loc = ploc("u_view_proj");   if (loc >= 0) pcp->uniform_matrix4fv(loc, glm::value_ptr(vp));
                loc = ploc("u_point_size");  if (loc >= 0) pcp->uniform1f(loc, point_size);
                loc = ploc("u_color_mode");  if (loc >= 0) pcp->uniform1i(loc, pc_color_mode);
                loc = ploc("u_num_leaves");   if (loc >= 0) pcp->uniform1ui(loc, GLuint(N));
                loc = ploc("u_rad_offset");   if (loc >= 0) pcp->uniform1ui(loc, GLuint(oct_interior));
                loc = ploc("u_rad_gain");     if (loc >= 0) pcp->uniform1f(loc, emissive_gain);
                loc = ploc("u_use_packed");   if (loc >= 0) pcp->uniform1i(loc, use_packed ? 1 : 0);
                loc = ploc("u_scene_min");   if (loc >= 0) pcp->uniform3f(loc, pp.scene_min.x, pp.scene_min.y, pp.scene_min.z);
                loc = ploc("u_scene_size");  if (loc >= 0) pcp->uniform3f(loc, pp.scene_size.x, pp.scene_size.y, pp.scene_size.z);
                loc = ploc("u_radius_scale"); if (loc >= 0) pcp->uniform1f(loc, pp.radius_scale);

                gl::enable(GL_PROGRAM_POINT_SIZE);
                gl::clear(GL_DEPTH_BUFFER_BIT);
                gl::enable(GL_DEPTH_TEST);
                gl::depth_func(GL_LESS);
                pc_vao.bind();
                gl::draw_arrays(GL_POINTS, 0, N);
                gl::disable(GL_PROGRAM_POINT_SIZE);
            }
        }

        // ---- 2c. Bounding-sphere wireframe overlay (S3) ----
        if (show_spheres && have_hierarchy) {
            gl::Program* sp = spherep->valid() ? spherep.get() : nullptr;
            if (sp) {
                sp->use();
                auto sloc = [&](const char* n) { return sp->uniform_location(n); };
                GLint loc;
                loc = sloc("u_view_proj"); if (loc >= 0) sp->uniform_matrix4fv(loc, glm::value_ptr(vp));
                loc = sloc("u_color");     if (loc >= 0) sp->uniform4fv(loc, glm::value_ptr(sphere_color));

                gl::enable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                gl::enable(GL_DEPTH_TEST);
                gl::depth_func(GL_LESS);

                int inst_first = 0, inst_count = oct_total_nodes;
                if (sphere_lod == 1) { inst_first = 0; inst_count = oct_interior; }
                else if (sphere_lod == 2) { inst_first = oct_interior; inst_count = N; }

                glVertexArrayVertexBuffer(sphere_mesh.vao.handle(), 1,
                                          sphere_buf.handle(),
                                          inst_first * sizeof(glm::vec4),
                                          sizeof(glm::vec4));
                sphere_mesh.vao.bind();
                glDrawElementsInstanced(GL_TRIANGLES, sphere_mesh.index_count,
                                        GL_UNSIGNED_INT, nullptr, inst_count);

                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                gl::disable(GL_BLEND);
            }
        }

        // ---- 3. ImGui ----
        gui.begin_frame();
        {
            ImGui::Begin("Point-Sampled Emissive Lighting");
            ImGui::Text("FPS: %.1f   Frame: %.2f ms", 1.0f / std::max(dt, 1e-6f), dt * 1000.0f);
            ImGui::Text("Resolution: %d x %d", targets.w, targets.h);
            ImGui::Separator();

            ImGui::Combo("Model", &model_index, "Cornell Box\0Sponza\0\0");
            if (ImGui::Button("Load")) {
                load_model(model_index);
                build_hierarchy();
            }
            ImGui::SameLine();
            ImGui::TextWrapped("Orbit: RMB drag  |  Zoom: scroll  |  Move: WASD");

            ImGui::Separator();
            ImGui::Combo("G-Buffer View", &view_mode, "Position\0Normal\0Albedo\0Emissive\0Depth\0Indirect\0");
            ImGui::SliderFloat("Exposure", &exposure, 0.05f, 5.0f);
            ImGui::SliderFloat("Gamma", &gamma, 1.0f, 3.0f);

            ImGui::Separator();
            ImGui::Text("S3 — Surfel Octree (N=%d, depth=%d)", N,
                        N > 0 ? int(std::ceil(std::log(float(N)) / std::log(8.0f))) : 0);
            static int last_budget_index = point_budget_index;
            ImGui::SliderInt("Point budget (log8)", &point_budget_index, 0, kBudgetCount - 1, "%d");
            if (point_budget_index != last_budget_index) {
                last_budget_index = point_budget_index;
                build_hierarchy();
            }
            if (ImGui::Button("Force rebuild")) build_hierarchy();
            ImGui::SameLine();
            ImGui::Text("lr=%.5f", leaf_radius);
            ImGui::Checkbox("Show point cloud", &show_points);
            if (show_points) {
                ImGui::Combo("Point color", &pc_color_mode, "Albedo\0Emissive\0Normal\0Position\0Radiance\0");
                ImGui::SliderFloat("Point size", &point_size, 1.0f, 20.0f);
                ImGui::Checkbox("Packed records (S3b)", &use_packed);
            }
            ImGui::Checkbox("Show bounding spheres", &show_spheres);
            if (show_spheres) {
                ImGui::Combo("Sphere LOD", &sphere_lod, "All\0Interior\0Leaves\0");
                ImGui::ColorEdit4("Sphere color", &sphere_color.x, ImGuiColorEditFlags_NoInputs);
            }

            ImGui::Separator();
            ImGui::Text("S4 — GPU Refit + Radiance");
            ImGui::Checkbox("Run refit", &run_refit);
            ImGui::SameLine();
            ImGui::Checkbox("Run radiance", &run_radiance);
            ImGui::SliderFloat("Emissive gain", &emissive_gain, 0.0f, 20.0f);
            ImGui::Combo("Bounces", &num_bounces, "0\01\02\0");
            ImGui::SameLine();
            ImGui::TextDisabled("(O(N^2) re-emission)");
            if (ImGui::Button("Validate S4 vs CPU")) validate_s4();
            ImGui::SameLine();
            ImGui::TextWrapped("(readback; compares pos/sphere/radiance)");

            ImGui::Separator();
            ImGui::Text("S5 — Micro-render Gather");
            ImGui::Checkbox("Run micro-render", &run_micro_render);
            static const int kMicroSizes[] = {8, 16, 24};
            int micro_idx = (micro_size == 24) ? 2 : (micro_size == 16 ? 1 : 0);
            if (ImGui::Combo("Micro size", &micro_idx, "8\0" "16\0" "24\0" "\0"))
                micro_size = kMicroSizes[micro_idx];
            static const int kScales[] = {1, 2, 4, 16};
            int scale_idx = (gather_scale >= 16) ? 3 : (gather_scale >= 4 ? 2 : (gather_scale >= 2 ? 1 : 0));
            if (ImGui::Combo("Gather res", &scale_idx, "Full\0" "1/2\0" "1/4\0" "1/16\0" "\0"))
                gather_scale = kScales[scale_idx];
            ImGui::SliderFloat("Micro gain", &micro_gain, 0.05f, 20.0f);
            if (gather_scale > 1) {
                ImGui::SliderFloat("Bilateral depth sigma", &bilateral_depth_sigma, 0.001f, 0.5f, "%.3f");
                ImGui::SliderFloat("Bilateral normal exp", &bilateral_normal_exp, 1.0f, 128.0f, "%.0f");
            }
            if (ImGui::Button("Indirect stats")) validate_s5();
            ImGui::SameLine();
            ImGui::TextWrapped("(readback: mean/max of indirect texture)");

            ImGui::Separator();
            ImGui::Text("GPU timings (ms)");
            ImGui::Text("  g-buffer   %6.2f", t_gbuf.readback());
            ImGui::Text("  refit      %6.2f", t_refit.readback());
            ImGui::Text("  radiance   %6.2f", t_radiance.readback());
            ImGui::Text("  bounce     %6.2f", t_bounce.readback());
            ImGui::Text("  micro      %6.2f", t_micro.readback());
            ImGui::Text("  bilateral  %6.2f", t_upsample.readback());
            ImGui::Text("  display    %6.2f", t_display.readback());
            ImGui::TextWrapped("View Indirect in G-Buffer View; wall opposite the emissive panel should light up.");

            ImGui::Separator();
            ImGui::Text("Pipeline");
            ImGui::Text("  S1/S2 scaffold: G-buffer + display");
            ImGui::Text("  S3: surfel BVH8 octree + packing");
            ImGui::Text("  S4: GPU refit + emissive radiance");
            ImGui::Text("  S5: micro-render gather (this stage)");
            ImGui::Text("  S6+ : S-J fallback, shadow rays, composite (see implementation.md)");
            ImGui::End();
        }
        gui.render();

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}

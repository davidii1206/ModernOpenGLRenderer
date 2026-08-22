// Example 38 — Surfel Ray Tracing (Schaufler & Jensen 2000)
//
// "Ray Tracing Point Sampled Geometry" — the scene is represented as a surfel
// cloud (points with position/normal/albedo/radius) stored in a sparse uniform
// spatial grid built entirely on the GPU (counting sort + prefix sum + compact).
// Rays are marched through the grid with 3D DDA; the first surfel disk hit is
// found per the paper, and the surface is reconstructed by interpolating nearby
// surfels weighted by (r - d_i) (paper Eq. 1). Shading is a single point light
// with 1/d^2 falloff (no shadows); the mirror box (tallBox, roughness ~0) spawns
// a perfect specular reflection ray (1 bounce).
//
// The same grid is the intended acceleration structure for a future micro-
// rendering gather (rasterizing surfels into per-gather-point micro-buffers):
// a local cell lookup replaces the brute-force per-surfel splat.

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

constexpr float kPi      = 3.14159265358979f;
constexpr int   kBEST_K  = 24;
constexpr float kCOVERAGE = 1.4f;
constexpr uint32_t kSEED = 42;
constexpr int   kBudgetOptions[] = {4096, 16384, 65536};
constexpr int   kBudgetCount = 3;
constexpr int   kBlockSize = 256;
constexpr int   kScanBlocksMax = 1024;   // scan_blocks.comp single workgroup limit
constexpr int   kMaxGridVolume = kBlockSize * kScanBlocksMax;  // 262144

// ---------------------------------------------------------------------------
// Triangle + surfel structures (CPU-side scene preparation)
// ---------------------------------------------------------------------------

struct Tri {
    glm::vec3 p[3];
    glm::vec3 n[3];
    float area = 0.0f;
    glm::vec3 alb{0.0f};
};

struct SamplePoint {
    glm::vec3 pos{0.0f};
    glm::vec3 nrm{0.0f};
    glm::vec3 alb{0.0f};
    float radius = 0.0f;   // adaptive per-surfel disk radius (paper future-work: local density)
    bool is_mirror = false;
};

// ---------------------------------------------------------------------------
// GPU timer (ping-pong query pair)
// ---------------------------------------------------------------------------

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
// G-buffer targets: direct-shaded color + position / normal / albedo(+mirror)
// ---------------------------------------------------------------------------

struct FrameTargets {
    int w = 0, h = 0;
    gl::Texture direct{gl::TextureType::tex_2d};
    gl::Texture position{gl::TextureType::tex_2d};
    gl::Texture normal{gl::TextureType::tex_2d};
    gl::Texture albedo{gl::TextureType::tex_2d};
    gl::Renderbuffer depth_rbo;
    gl::Framebuffer fbo;
};

void create_gbuffer(FrameTargets& t, int w, int h) {
    t.w = w;
    t.h = h;

    auto tex2 = [&](gl::Texture& tex, GLenum internal) {
        tex = gl::Texture{gl::TextureType::tex_2d};
        tex.image_2d(0, internal, w, h,
                     internal == GL_RGBA8 ? GL_RGBA : GL_RGBA,
                     GL_FLOAT, nullptr, 1);
        tex.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        tex.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        tex.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        tex.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };
    tex2(t.direct,   GL_RGBA16F);
    tex2(t.position, GL_RGBA16F);
    tex2(t.normal,   GL_RGBA16F);
    tex2(t.albedo,   GL_RGBA8);

    t.depth_rbo = gl::Renderbuffer{};
    t.depth_rbo.storage(GL_DEPTH_COMPONENT24, w, h);

    t.fbo.bind();
    t.fbo.attach_texture(GL_COLOR_ATTACHMENT0, t.direct);
    t.fbo.attach_texture(GL_COLOR_ATTACHMENT1, t.position);
    t.fbo.attach_texture(GL_COLOR_ATTACHMENT2, t.normal);
    t.fbo.attach_texture(GL_COLOR_ATTACHMENT3, t.albedo);
    t.fbo.attach_renderbuffer(GL_DEPTH_ATTACHMENT, t.depth_rbo);
    GLenum bufs[4] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
                      GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
    glDrawBuffers(4, bufs);
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
// Triangle extraction (bakes model_mat into world space)
// ---------------------------------------------------------------------------

struct ExtractedMesh {
    std::vector<Tri> tris;
    std::vector<bool> tri_mirror;   // parallel to tris
};

ExtractedMesh extract_triangles(const gfx::Model& model, const glm::mat4& model_mat) {
    ExtractedMesh out;
    glm::mat3 normal_mat = glm::transpose(glm::inverse(glm::mat3(model_mat)));

    for (size_t mi = 0; mi < model.mesh_count(); ++mi) {
        const gfx::Mesh& mesh = model.mesh(int(mi));
        int mati = model.mesh_material(int(mi));
        const auto& mat = model.material_info(size_t(mati >= 0 ? mati : 0));
        glm::vec3 alb(mat.base_color_factor[0], mat.base_color_factor[1], mat.base_color_factor[2]);
        bool is_mirror = (mat.name.find("tallBox") != std::string::npos) ||
                         (mat.roughness_factor < 0.05f);
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
            out.tris.push_back(t);
            out.tri_mirror.push_back(is_mirror);
        };

        if (is.empty()) {
            for (size_t v = 0; v + 2 < vs.size(); v += 3)
                add_tri(vs[v], vs[v + 1], vs[v + 2]);
        } else {
            for (size_t k = 0; k + 2 < is.size(); k += 3)
                add_tri(vs[is[k]], vs[is[k + 1]], vs[is[k + 2]]);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Mitchell best-candidate sampling, area-weighted
// ---------------------------------------------------------------------------

SamplePoint sample_on_tri(const Tri& tri, std::mt19937& rng, bool is_mirror) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float su = dist(rng), sv = dist(rng);
    if (su + sv > 1.0f) { su = 1.0f - su; sv = 1.0f - sv; }
    float sw = 1.0f - su - sv;
    SamplePoint s;
    s.pos = su * tri.p[0] + sv * tri.p[1] + sw * tri.p[2];
    s.nrm = glm::normalize(su * tri.n[0] + sv * tri.n[1] + sw * tri.n[2]);
    s.alb = tri.alb;
    s.is_mirror = is_mirror;
    return s;
}

// Returns (surfels, total_area).
std::pair<std::vector<SamplePoint>, float> sample_points_mitchell(
    const std::vector<Tri>& tris, const std::vector<bool>& tri_mirror,
    float total_area, int N, int K)
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
    float s = std::max(std::sqrt(total_area / float(N)), 1e-6f);

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
            float bound = std::max(stop_at, float(R * R) * s * s);
            if (any && d2 <= bound) break;
        }
        return d2 < 1e29f;
    };

    {
        int ti = tri_pick(rng);
        accepted.push_back(sample_on_tri(tris[ti], rng, tri_mirror[ti]));
        add_point(0);
    }

    for (int i = 1; i < N; ++i) {
        float best_dist = -1.0f;
        SamplePoint best_sp{};
        for (int k = 0; k < K; ++k) {
            float r = unit(rng) * total_area;
            int ti = int(std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin());
            ti = std::clamp(ti, 0, int(tris.size()) - 1);
            SamplePoint cand = sample_on_tri(tris[ti], rng, tri_mirror[ti]);
            float min_d;
            if (!min_dist_to(cand.pos, best_dist, min_d)) min_d = 1e30f;
            if (min_d > best_dist) { best_dist = min_d; best_sp = cand; }
        }
        accepted.push_back(best_sp);
        add_point(i);
    }

    // Per-surfel adaptive radius (paper §7 future work): shrink the disk radius
    // where the local sampling is denser, i.e. near edges/corners where surfaces
    // converge, so disks don't protrude past geometric edges (crisper silhouette).
    // Local tile area ~ nn^2 => radius = kCOVERAGE * nn / sqrt(pi), which matches the
    // global leaf_radius on flat, evenly sampled faces and shrinks at edges.
    const float global_r = kCOVERAGE * std::sqrt(total_area / (float(N) * kPi));
    const float nn_scale = kCOVERAGE / std::sqrt(kPi);
    // Mild shrink only: keep disks mostly overlapping so the surface stays
    // continuous (no gaps). Edge sharpness is handled by the per-surfel cut
    // planes, so we no longer need an aggressive radius reduction here.
    constexpr float kMinFrac = 0.85f;
    for (int i = 0; i < N; ++i) {
        glm::ivec3 c0 = cell_of(accepted[i].pos);
        float nn2 = 1e30f;
        for (int R = 0; R <= kMaxRing; ++R) {
            bool any = false;
            for (int dx = -R; dx <= R; ++dx)
            for (int dy = -R; dy <= R; ++dy)
            for (int dz = -R; dz <= R; ++dz) {
                if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != R) continue;
                auto it = grid.find(cell_key(c0 + glm::ivec3(dx, dy, dz)));
                if (it == grid.end()) continue;
                any = true;
                for (int j : it->second) {
                    if (j == i) continue;
                    float t = glm::dot(accepted[i].pos - accepted[j].pos,
                                       accepted[i].pos - accepted[j].pos);
                    if (t < nn2) nn2 = t;
                }
            }
            if (any && nn2 < 1e29f) {
                float bound = float(R * R) * s * s;
                if (nn2 <= bound) break;
            }
        }
        float r = (nn2 < 1e29f) ? nn_scale * std::sqrt(nn2) : global_r;
        r = std::clamp(r, kMinFrac * global_r, global_r);
        accepted[i].radius = r;
    }
    return {accepted, total_area};
}

// ---------------------------------------------------------------------------
// Surfel edge cutting: for each surfel, store up to K cut planes so the shader
// can clip its disk to the sharp-edge boundaries (no surfel-pair adjacency).
//
// A surfel on face A near a sharp edge against face B stores B's plane as a cut:
//   cut.xyz = n_B,  cut.w = dot(pos_B - pos_A, n_B) >= 0
// with the clip condition  dot(q - pos_A, n_B) <= cut.w  (keeps A's side of B's
// plane). Surfels with fewer than K cuts pad with cut.w = +inf (never clipped).
// A box-corner surfel touches up to 2-3 sharp edges, so K covers it.
// ---------------------------------------------------------------------------

constexpr int kSurfelCuts = 4;
constexpr float kCutInf = 1e30f;

std::vector<glm::vec4> compute_surfel_cuts(const std::vector<SamplePoint>& pts,
                                           float search_radius) {
    const int N = int(pts.size());
    std::vector<glm::vec4> cuts(size_t(N) * kSurfelCuts, glm::vec4(0.0f, 0.0f, 0.0f, kCutInf));

    // Binning grid over the scene AABB at cell ~ search radius.
    glm::vec3 mn(1e30f), mx(-1e30f);
    for (const auto& s : pts) { mn = glm::min(mn, s.pos); mx = glm::max(mx, s.pos); }
    glm::vec3 sz = mx - mn;
    float cell = std::max(search_radius, 1e-4f);
    glm::ivec3 res(std::max(1, int(std::ceil(sz.x / cell))),
                   std::max(1, int(std::ceil(sz.y / cell))),
                   std::max(1, int(std::ceil(sz.z / cell))));

    auto cell_of = [&](const glm::vec3& p) -> glm::ivec3 {
        return glm::ivec3(glm::floor((p - mn) / cell));
    };
    auto cell_key = [](const glm::ivec3& c) -> uint64_t {
        return (uint64_t(uint32_t(c.x) & 0x1FFFFF) << 42) |
               (uint64_t(uint32_t(c.y) & 0x1FFFFF) << 21) |
                uint64_t(uint32_t(c.z) & 0x1FFFFF);
    };
    std::unordered_map<uint64_t, std::vector<int>> grid;
    for (int i = 0; i < N; ++i)
        grid[cell_key(cell_of(pts[i].pos))].push_back(i);

    constexpr float kSharpDot = 0.6f;   // |n_i . n_j| < this => sharp edge
    for (int i = 0; i < N; ++i) {
        glm::ivec3 c0 = cell_of(pts[i].pos);
        int ncuts = 0;
        // Sweep a small neighborhood; a disk of radius ~ search_radius on A can
        // cross edges within a few cells. Ring 0..2 of a cell-sized grid is enough.
        for (int dz = -2; dz <= 2 && ncuts < kSurfelCuts; ++dz)
        for (int dy = -2; dy <= 2 && ncuts < kSurfelCuts; ++dy)
        for (int dx = -2; dx <= 2 && ncuts < kSurfelCuts; ++dx) {
            auto it = grid.find(cell_key(c0 + glm::ivec3(dx, dy, dz)));
            if (it == grid.end()) continue;
            for (int j : it->second) {
                if (j == i) continue;
                const SamplePoint& a = pts[i];
                const SamplePoint& b = pts[j];
                // Only across a sharp edge (different face orientation) and close.
                float nd = glm::dot(a.nrm, b.nrm);
                if (std::abs(nd) > kSharpDot) continue;
                if (glm::length(a.pos - b.pos) > 2.0f * search_radius) continue;

                // Clip against b's plane, keeping a's side.
                glm::vec3 nb = b.nrm;
                float off = glm::dot(b.pos - a.pos, nb);
                if (off < 0.0f) { nb = -nb; off = -off; }   // force keep-side to -inf side

                // Deduplicate against existing cuts for this surfel.
                bool dup = false;
                for (int k = 0; k < ncuts; ++k) {
                    glm::vec4& ex = cuts[size_t(i) * kSurfelCuts + k];
                    if (std::abs(glm::dot(glm::vec3(ex), nb)) > 0.98f && std::abs(ex.w - off) < 1e-3f) {
                        dup = true; break;
                    }
                }
                if (dup) continue;
                cuts[size_t(i) * kSurfelCuts + ncuts] = glm::vec4(nb, off);
                ++ncuts;
                if (ncuts >= kSurfelCuts) break;
            }
        }
    }
    return cuts;
}

// ---------------------------------------------------------------------------
// Disk cache for the surfel cloud (keyed on model stamp / material hash /
// transform / N / seed / coverage). The GPU grid is rebuilt on demand.
// ---------------------------------------------------------------------------

constexpr uint32_t kCacheMagic   = 0x38535254;  // "8SRT"
constexpr uint32_t kCacheVersion = 3;

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
        h = (h ^ uint64_t(std::bit_cast<uint32_t>(m.metallic_factor))) * 1099511628211ull;
        h = (h ^ uint64_t(std::bit_cast<uint32_t>(m.roughness_factor))) * 1099511628211ull;
    }
    return h;
}

uint64_t matrix_hash(const glm::mat4& m) {
    uint64_t h = 1469598103934665603ull;
    const float* p = glm::value_ptr(m);
    for (int i = 0; i < 16; ++i)
        h = (h ^ uint64_t(std::bit_cast<uint32_t>(p[i]))) * 1099511628211ull;
    return h;
}

struct CacheKey {
    uint64_t model_stamp = 0;
    uint64_t mat_hash = 0;
    uint64_t transform_hash = 0;
    uint32_t N = 0;
    uint32_t best_k = kBEST_K;
    float coverage = kCOVERAGE;
    uint32_t seed = kSEED;
};

std::string cache_path_for(const std::string& model_path, int N) {
    std::filesystem::path p(model_path);
    return "cache/38_" + p.stem().string() + "_N" + std::to_string(N) + ".bin";
}

bool save_cache(const std::string& path, const CacheKey& key, const std::vector<SamplePoint>& pts,
                float leaf_radius, float total_area) {
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
    wf(leaf_radius);
    wf(total_area);
    f.write(reinterpret_cast<const char*>(pts.data()), pts.size() * sizeof(SamplePoint));
    return f.good();
}

bool load_cache(const std::string& path, const CacheKey& key, std::vector<SamplePoint>& pts,
                float& leaf_radius, float& total_area) {
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

    leaf_radius = rf();
    total_area = rf();
    pts.resize(key.N);
    f.read(reinterpret_cast<char*>(pts.data()), key.N * sizeof(SamplePoint));
    return f.good();
}

// ---------------------------------------------------------------------------
// Unit-cube wireframe mesh for occupied-grid-cell debug overlay
// ---------------------------------------------------------------------------

struct CubeMesh {
    gl::VertexArray vao;
    gl::Buffer vbo, ebo;
    int index_count = 0;
};

CubeMesh create_wireframe_cube() {
    CubeMesh cm;
    const float s = 1.0f;
    glm::vec3 verts[8] = {
        {-s, -s, -s}, { s, -s, -s}, { s,  s, -s}, {-s,  s, -s},
        {-s, -s,  s}, { s, -s,  s}, { s,  s,  s}, {-s,  s,  s},
    };
    uint32_t idx[24] = {
        0,1, 1,2, 2,3, 3,0,
        4,5, 5,6, 6,7, 7,4,
        0,4, 1,5, 2,6, 3,7,
    };
    cm.index_count = 24;
    cm.vbo = gl::Buffer(gl::BufferType::vertex, gl::BufferUsage::static_draw);
    cm.vbo.data(verts, sizeof(verts));
    cm.ebo = gl::Buffer(gl::BufferType::index, gl::BufferUsage::static_draw);
    cm.ebo.data(idx, sizeof(idx));
    return cm;
}

void setup_cube_vao(CubeMesh& cm, const gl::Buffer& instance_buf) {
    cm.vao.bind();
    cm.vbo.bind();
    cm.vao.attrib_pointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (const void*)0);
    cm.vao.enable_attrib(0);
    instance_buf.bind();
    cm.vao.attrib_pointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (const void*)0);
    cm.vao.enable_attrib(1);
    glVertexAttribDivisor(1, 1);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cm.ebo.handle());
    gl::VertexArray::unbind();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    gllib::log_to_stderr(gllib::LogLevel::info);
    gfx::Window window({"38 Surfel Ray Tracing", 1600, 900});
    window.vsync(false);

    gfx::ImGuiOverlay gui;
    if (!gui.init(window)) {
        gllib::log(gllib::LogLevel::error, "ImGui init failed");
        return EXIT_FAILURE;
    }

    // Hot-reloadable shaders.
    gl::HotReloadProgram display_prog;
    display_prog.add_stage("shaders/display.vert", gl::ShaderType::vertex);
    display_prog.add_stage("shaders/display.frag", gl::ShaderType::fragment);
    gl::HotReloadProgram gbuf_prog;
    gbuf_prog.add_stage("shaders/gbuf.vert", gl::ShaderType::vertex);
    gbuf_prog.add_stage("shaders/gbuf.frag", gl::ShaderType::fragment);
    gl::HotReloadProgram raytrace_prog;
    raytrace_prog.add_stage("shaders/raytrace.comp", gl::ShaderType::compute);
    gl::HotReloadProgram pc_prog;
    pc_prog.add_stage("shaders/pc.vert", gl::ShaderType::vertex);
    pc_prog.add_stage("shaders/pc.frag", gl::ShaderType::fragment);
    gl::HotReloadProgram box_prog;
    box_prog.add_stage("shaders/grid_box.vert", gl::ShaderType::vertex);
    box_prog.add_stage("shaders/grid_box.frag", gl::ShaderType::fragment);
    gl::HotReloadProgram count_prog;
    count_prog.add_stage("shaders/grid_count.comp", gl::ShaderType::compute);
    gl::HotReloadProgram scan_block_prog;
    scan_block_prog.add_stage("shaders/scan_block.comp", gl::ShaderType::compute);
    gl::HotReloadProgram scan_blocks_prog;
    scan_blocks_prog.add_stage("shaders/scan_blocks.comp", gl::ShaderType::compute);
    gl::HotReloadProgram scan_finish_prog;
    scan_finish_prog.add_stage("shaders/scan_finish.comp", gl::ShaderType::compute);
    gl::HotReloadProgram compact_prog;
    compact_prog.add_stage("shaders/grid_compact.comp", gl::ShaderType::compute);

    display_prog.poll();
    gbuf_prog.poll();
    raytrace_prog.poll();
    pc_prog.poll();
    box_prog.poll();
    count_prog.poll();
    scan_block_prog.poll();
    scan_blocks_prog.poll();
    scan_finish_prog.poll();
    compact_prog.poll();

    auto display = display_prog.take_program();
    auto gbuf = gbuf_prog.take_program();
    auto raytrace = raytrace_prog.take_program();
    auto pc = pc_prog.take_program();
    auto boxp = box_prog.take_program();
    auto count_cs = count_prog.take_program();
    auto scan_block_cs = scan_block_prog.take_program();
    auto scan_blocks_cs = scan_blocks_prog.take_program();
    auto scan_finish_cs = scan_finish_prog.take_program();
    auto compact_cs = compact_prog.take_program();
    if (!display || !gbuf || !raytrace || !pc || !boxp ||
        !count_cs || !scan_block_cs || !scan_blocks_cs || !scan_finish_cs || !compact_cs)
        return EXIT_FAILURE;

    // Fullscreen triangle for display.
    gl::VertexArray fsq_vao;
    gl::VertexArray pc_vao;   // empty VAO; point cloud reads SSBOs via gl_VertexID

    // Camera.
    gfx::Camera cam;
    cam.perspective(45.0f, float(window.framebuffer_width()) / float(window.framebuffer_height()),
                    0.1f, 1000.0f);
    glm::vec3 cam_pos(0.0f, 0.5f, 4.5f);
    glm::vec3 cam_tgt(0.0f, 0.5f, 0.0f);
    if (getenv("SRT_CAM_X")) cam_pos.x = std::atof(getenv("SRT_CAM_X"));
    if (getenv("SRT_CAM_Y")) cam_pos.y = std::atof(getenv("SRT_CAM_Y"));
    if (getenv("SRT_CAM_Z")) cam_pos.z = std::atof(getenv("SRT_CAM_Z"));
    if (getenv("SRT_TGT_X")) cam_tgt.x = std::atof(getenv("SRT_TGT_X"));
    if (getenv("SRT_TGT_Y")) cam_tgt.y = std::atof(getenv("SRT_TGT_Y"));
    if (getenv("SRT_TGT_Z")) cam_tgt.z = std::atof(getenv("SRT_TGT_Z"));
    cam.look_at(cam_pos, cam_tgt);

    // Model.
    const char* model_path = "CornellBoxMirrorBox.glb";
    gfx::Model model;
    if (!model.load(model_path)) {
        gllib::log(gllib::LogLevel::error, "Failed to load CornellBoxMirrorBox.glb");
        return EXIT_FAILURE;
    }
    // The model node rotates 90 deg about X; gfx::Model keeps meshes in local
    // space, so bake it here (makes the box axis-aligned: +Y up, open face -Z).
    glm::mat4 model_mat = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));

    gllib::logf(gllib::LogLevel::info, "Loaded: %zu meshes, %zu materials",
                model.mesh_count(), model.material_count());
    for (size_t i = 0; i < model.material_count(); ++i) {
        const auto& m = model.material_info(i);
        gllib::logf(gllib::LogLevel::info, "mat %zu '%s': albedo(%.3f %.3f %.3f) rough=%.3f",
                    i, m.name.c_str(), m.base_color_factor[0], m.base_color_factor[1],
                    m.base_color_factor[2], m.roughness_factor);
    }

    // ---- Surfel cloud (CPU, cached) ----
    int point_budget_index = 1;   // default N=16384
    int N = 0;
    float leaf_radius = 0.0f;     // nominal (flat-face) radius
    float search_radius = 0.0f;   // max per-surfel radius, for grid/traversal bounds
    float total_area = 0.0f;
    std::vector<SamplePoint> surfels;

    auto build_surfels = [&]() {
        N = kBudgetOptions[point_budget_index];
        CacheKey key;
        key.model_stamp = file_stamp(model_path, nullptr);
        key.N = uint32_t(N);
        key.mat_hash = material_hash(model);
        key.transform_hash = matrix_hash(model_mat);

        std::string cpath = cache_path_for(model_path, N);
        if (load_cache(cpath, key, surfels, leaf_radius, total_area)) {
            gllib::logf(gllib::LogLevel::info, "cache hit %s (N=%d)", cpath.c_str(), N);
        } else {
            auto t0 = std::chrono::steady_clock::now();
            ExtractedMesh em = extract_triangles(model, model_mat);
            total_area = 0.0f;
            for (const auto& t : em.tris) total_area += t.area;

            auto [pts, ta] = sample_points_mitchell(em.tris, em.tri_mirror, total_area, N, kBEST_K);
            surfels = std::move(pts);
            leaf_radius = kCOVERAGE * std::sqrt(total_area / (float(N) * kPi));
            if (save_cache(cpath, key, surfels, leaf_radius, total_area))
                gllib::logf(gllib::LogLevel::info, "cached to %s", cpath.c_str());
            auto t1 = std::chrono::steady_clock::now();
            gllib::logf(gllib::LogLevel::info,
                        "sampled %d surfels in %.0f ms (%.1f tris)",
                        N, std::chrono::duration<double, std::milli>(t1 - t0).count(),
                        double(em.tris.size()));
        }
    };
    build_surfels();

    // Max per-surfel radius bounds the grid cell size and ray-traversal search
    // radius (must cover the largest disk so neighborhood windows reach it).
    search_radius = 0.0f;
    for (const auto& s : surfels) search_radius = std::max(search_radius, s.radius);

    // Scene AABB from surfels.
    glm::vec3 mn(1e30f), mx(-1e30f);
    for (const auto& s : surfels) {
        mn = glm::min(mn, s.pos);
        mx = glm::max(mx, s.pos);
    }
    glm::vec3 scene_size = mx - mn;
    gllib::logf(gllib::LogLevel::info,
                "surfel AABB (%.3f %.3f %.3f)-(%.3f %.3f %.3f), radius=%.5f, total_area=%.3f",
                mn.x, mn.y, mn.z, mx.x, mx.y, mx.z, leaf_radius, total_area);

    // ---- GPU buffers ----
    gl::Buffer surfel_pos_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer surfel_nrm_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer surfel_alb_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer surfel_cut_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer cell_count_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer cell_start_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer packed_idx_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer surfel_slot_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer block_sum_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);

    std::vector<glm::vec4> vp(N), vn(N), va(N);
    for (int i = 0; i < N; ++i) {
        vp[i] = glm::vec4(surfels[i].pos, surfels[i].radius);
        vn[i] = glm::vec4(surfels[i].nrm, 0.0f);
        va[i] = glm::vec4(surfels[i].alb, surfels[i].is_mirror ? 1.0f : 0.0f);
    }
    surfel_pos_buf.data(vp.data(), vp.size() * sizeof(glm::vec4));
    surfel_nrm_buf.data(vn.data(), vn.size() * sizeof(glm::vec4));
    surfel_alb_buf.data(va.data(), va.size() * sizeof(glm::vec4));
    std::vector<glm::vec4> cuts = compute_surfel_cuts(surfels, search_radius);
    surfel_cut_buf.data(cuts.data(), cuts.size() * sizeof(glm::vec4));
    surfel_slot_buf.data(nullptr, size_t(N) * sizeof(uint32_t));
    packed_idx_buf.data(nullptr, size_t(N) * sizeof(uint32_t));

    // Grid parameters. Cell size must fit the largest disk so every surfel is
    // reachable from its own cell's neighborhood.
    float cell_size = 2.0f * search_radius;
    glm::ivec3 grid_res(0);
    uint32_t grid_volume = 0;

    auto compute_grid_params = [&](float cs) {
        for (int a = 0; a < 3; ++a)
            grid_res[a] = std::max(1, int(std::ceil(scene_size[a] / cs)));
        // Enforce volume cap for the single-workgroup scan.
        while (grid_res.x * grid_res.y * grid_res.z > kMaxGridVolume) {
            cs *= 2.0f;
            for (int a = 0; a < 3; ++a)
                grid_res[a] = std::max(1, int(std::ceil(scene_size[a] / cs)));
        }
        grid_volume = uint32_t(grid_res.x * grid_res.y * grid_res.z);
        cell_size = cs;
    };
    compute_grid_params(cell_size);

    cell_count_buf.data(nullptr, size_t(grid_volume) * sizeof(uint32_t));
    cell_start_buf.data(nullptr, size_t(grid_volume) * sizeof(uint32_t));
    block_sum_buf.data(nullptr, kScanBlocksMax * sizeof(uint32_t));

    const uint32_t zero = 0;

    // ---- Grid build (GPU) ----
    auto set_grid_uniforms = [&](gl::Program& prog) {
        auto loc = [&](const char* n) { return prog.uniform_location(n); };
        GLint l;
        l = loc("u_grid_min");    if (l >= 0) prog.uniform3f(l, mn.x, mn.y, mn.z);
        l = loc("u_inv_cell");    if (l >= 0) prog.uniform1f(l, 1.0f / cell_size);
        GLint gr[3] = {grid_res.x, grid_res.y, grid_res.z};
        l = loc("u_grid_res");    if (l >= 0) prog.uniform3iv(l, gr);
        l = loc("u_num_surfels"); if (l >= 0) prog.uniform1ui(l, GLuint(N));
    };

    auto bind_count = [&] {
        surfel_pos_buf.bind_base(0);
        cell_count_buf.bind_base(1);
        surfel_slot_buf.bind_base(2);
    };
    auto bind_scan_block = [&] {
        cell_count_buf.bind_base(0);
        cell_start_buf.bind_base(1);
        block_sum_buf.bind_base(2);
    };
    auto bind_scan_finish = [&] {
        cell_start_buf.bind_base(0);
        block_sum_buf.bind_base(1);
    };
    auto bind_compact = [&] {
        surfel_pos_buf.bind_base(0);
        surfel_slot_buf.bind_base(1);
        cell_start_buf.bind_base(2);
        packed_idx_buf.bind_base(3);
    };
    auto bind_rt = [&] {
        surfel_pos_buf.bind_base(0);
        surfel_nrm_buf.bind_base(1);
        surfel_alb_buf.bind_base(2);
        cell_start_buf.bind_base(3);
        cell_count_buf.bind_base(4);
        packed_idx_buf.bind_base(5);
        surfel_cut_buf.bind_base(6);
    };

    auto build_grid = [&]() {
        // Reset cell_count.
        cell_count_buf.clear(GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);

        // Pass 1: count + slot.
        bind_count();
        count_cs->use();
        set_grid_uniforms(*count_cs);
        gl::dispatch_compute((N + kBlockSize - 1) / kBlockSize, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Pass 2: prefix sum (block scan -> block sums scan -> finish).
        uint32_t nblocks = (grid_volume + kBlockSize - 1) / kBlockSize;
        bind_scan_block();
        scan_block_cs->use();
        {
            GLint l = scan_block_cs->uniform_location("u_n");
            if (l >= 0) scan_block_cs->uniform1ui(l, grid_volume);
        }
        gl::dispatch_compute(nblocks, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        block_sum_buf.bind_base(0);
        scan_blocks_cs->use();
        {
            GLint l = scan_blocks_cs->uniform_location("u_nblocks");
            if (l >= 0) scan_blocks_cs->uniform1ui(l, nblocks);
        }
        gl::dispatch_compute(1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        bind_scan_finish();
        scan_finish_cs->use();
        {
            GLint l = scan_finish_cs->uniform_location("u_n");
            if (l >= 0) scan_finish_cs->uniform1ui(l, grid_volume);
        }
        gl::dispatch_compute(nblocks, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Pass 3: compact.
        bind_compact();
        compact_cs->use();
        set_grid_uniforms(*compact_cs);
        gl::dispatch_compute((N + kBlockSize - 1) / kBlockSize, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        bind_rt();   // restore raytrace bindings
    };

    bind_rt();
    build_grid();

    // Occupied-cell AABB instance buffer for the debug overlay.
    gl::Buffer occ_aabb_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    std::vector<glm::vec4> occ_aabbs;
    auto refresh_occupied_cells = [&]() {
        occ_aabbs.clear();
        std::vector<uint32_t> counts(grid_volume);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();
        void* ptr = cell_count_buf.map_range(0, grid_volume * sizeof(uint32_t), GL_MAP_READ_BIT);
        if (ptr) {
            std::memcpy(counts.data(), ptr, grid_volume * sizeof(uint32_t));
            cell_count_buf.unmap();
        }
        glm::ivec3 res = grid_res;
        for (uint32_t ci = 0; ci < grid_volume; ++ci) {
            if (counts[ci] == 0) continue;
            int x = int(ci % uint32_t(res.x));
            int y = int((ci / uint32_t(res.x)) % uint32_t(res.y));
            int z = int(ci / (uint32_t(res.x) * uint32_t(res.y)));
            glm::vec3 cell_min = mn + glm::vec3(x, y, z) * cell_size;
            occ_aabbs.push_back(glm::vec4(cell_min, cell_size));
        }
        occ_aabb_buf.data(occ_aabbs.data(), occ_aabbs.size() * sizeof(glm::vec4));
        gllib::logf(gllib::LogLevel::info,
                    "grid: cell=%.4f res=%dx%dx%d vol=%u occupied=%zu (N=%d)",
                    cell_size, res.x, res.y, res.z, grid_volume, occ_aabbs.size(), N);
    };
    refresh_occupied_cells();

    // Ray trace output textures.
    gl::Texture color_tex{gl::TextureType::tex_2d};
    gl::Texture aux_tex{gl::TextureType::tex_2d};
    auto create_rt_tex = [&](int w, int h) {
        color_tex = gl::Texture{gl::TextureType::tex_2d};
        color_tex.image_2d(0, GL_RGBA16F, w, h, GL_RGBA, GL_FLOAT, nullptr, 1);
        color_tex.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        color_tex.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        color_tex.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        color_tex.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        aux_tex = gl::Texture{gl::TextureType::tex_2d};
        aux_tex.image_2d(0, GL_RGBA16F, w, h, GL_RGBA, GL_FLOAT, nullptr, 1);
        aux_tex.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        aux_tex.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        aux_tex.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        aux_tex.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };
    create_rt_tex(window.framebuffer_width(), window.framebuffer_height());

    // G-buffer (rasterized direct view + surfaces for reflection).
    FrameTargets gbuf_t;
    create_gbuffer(gbuf_t, window.framebuffer_width(), window.framebuffer_height());

    // Per-mesh mirror flag (for the raster G-buffer pass).
    auto mesh_is_mirror = [&](size_t mi) -> bool {
        int mati = model.mesh_material(int(mi));
        const auto& mat = model.material_info(size_t(mati >= 0 ? mati : 0));
        return (mat.name.find("tallBox") != std::string::npos) || (mat.roughness_factor < 0.05f);
    };

    // Debug overlays.
    CubeMesh cube_mesh = create_wireframe_cube();
    setup_cube_vao(cube_mesh, occ_aabb_buf);

    // ---- UI state ----
    bool traced = false;
    bool realtime = false;
    bool show_points = false;
    bool show_cells = false;
    int pc_color_mode = 0;
    float point_size = 3.0f;
    int view_mode = 0;
    if (getenv("SRT_VIEW")) view_mode = std::atoi(getenv("SRT_VIEW"));
    float exposure = 1.0f;
    float gamma = 2.2f;
    const float far_plane = 100.0f;
    int max_bounces = 1;
    if (getenv("SRT_BOUNCES")) max_bounces = std::atoi(getenv("SRT_BOUNCES"));
    float radius_scale = 1.25f;
    if (getenv("SRT_RADIUS")) radius_scale = std::atof(getenv("SRT_RADIUS"));
    float light_intensity = 6.0f;
    float ambient = 0.08f;
    glm::vec3 light_pos = mn + glm::vec3(scene_size.x * 0.5f, scene_size.y * 0.95f, scene_size.z * 0.5f);
    glm::vec3 light_color(1.0f, 0.98f, 0.92f);

    GpuTimer t_ray("raytrace");
    GpuTimer t_display("display");

    bool captured = false;
    double last = window.time();

    // Headless test hook: SRT_AUTOTRACE=1 runs the trace every frame, saves a
    // screenshot to SRT_SHOT (default rt38.png) on frame SRT_FRAME (default 5),
    // then exits. Used to verify the renderer without a GUI.
    bool autotrace = getenv("SRT_AUTOTRACE") != nullptr;
    if (autotrace) realtime = true;
    const char* shot_path = getenv("SRT_SHOT");
    int shot_frame = getenv("SRT_FRAME") ? std::atoi(getenv("SRT_FRAME")) : 5;
    uint64_t frame_counter = 0;

    // Rasterize the G-buffer: world pos/normal/albedo(+mirror) + direct shaded color.
    auto run_gbuffer = [&]() {
        int fw = window.framebuffer_width(), fh = window.framebuffer_height();
        glm::mat4 vp = cam.view_projection();

        gbuf_t.fbo.bind();
        gl::viewport(0, 0, gbuf_t.w, gbuf_t.h);
        gl::clear_color(0.0f, 0.0f, 0.0f, 0.0f);
        gl::clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        gl::enable(GL_DEPTH_TEST);
        gl::depth_func(GL_LESS);

        if (gbuf->valid()) {
            gbuf->use();
            auto loc = [&](const char* n) { return gbuf->uniform_location(n); };
            GLint l;
            l = loc("u_view_proj");      if (l >= 0) gbuf->uniform_matrix4fv(l, glm::value_ptr(vp));
            l = loc("u_model");           if (l >= 0) gbuf->uniform_matrix4fv(l, glm::value_ptr(model_mat));
            glm::mat3 normal_mat = glm::transpose(glm::inverse(glm::mat3(model_mat)));
            l = loc("u_normal_mat");      if (l >= 0) gbuf->uniform_matrix3fv(l, glm::value_ptr(normal_mat));
            l = loc("u_light_pos");       if (l >= 0) gbuf->uniform3f(l, light_pos.x, light_pos.y, light_pos.z);
            l = loc("u_light_color");     if (l >= 0) gbuf->uniform3f(l, light_color.x, light_color.y, light_color.z);
            l = loc("u_light_intensity"); if (l >= 0) gbuf->uniform1f(l, light_intensity);
            l = loc("u_ambient");         if (l >= 0) gbuf->uniform1f(l, ambient);

            for (size_t i = 0; i < model.mesh_count(); ++i) {
                int mi = model.mesh_material(int(i));
                const auto& mat = model.material_info(size_t(mi >= 0 ? mi : 0));
                l = loc("u_albedo");    if (l >= 0) gbuf->uniform3f(l, mat.base_color_factor[0], mat.base_color_factor[1], mat.base_color_factor[2]);
                l = loc("u_is_mirror"); if (l >= 0) gbuf->uniform1f(l, mesh_is_mirror(i) ? 1.0f : 0.0f);
                model.mesh(i).draw();
            }
        }
        gl::Framebuffer::unbind(gl::FramebufferType::both);
        gl::disable(GL_DEPTH_TEST);
    };

    // Composite: rasterized direct shading everywhere; surfel ray-traced mirror
    // reflections only on the mirror box (Schaufler & Jensen).
    auto run_raytrace = [&]() {
        int fw = window.framebuffer_width(), fh = window.framebuffer_height();
        t_ray.begin();
        gbuf_t.direct.bind(0);
        gbuf_t.position.bind(1);
        gbuf_t.normal.bind(2);
        gbuf_t.albedo.bind(3);
        color_tex.bind_image(3, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        aux_tex.bind_image(4, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

        raytrace->use();
        auto loc = [&](const char* n) { return raytrace->uniform_location(n); };
        GLint l;
        GLint isz[2] = {fw, fh};
        l = loc("u_img_size");     if (l >= 0) raytrace->uniform2iv(l, isz);
        l = loc("u_cam_pos");      if (l >= 0) raytrace->uniform3f(l, cam.position().x, cam.position().y, cam.position().z);
        l = loc("u_far");          if (l >= 0) raytrace->uniform1f(l, far_plane);
        l = loc("u_grid_min");     if (l >= 0) raytrace->uniform3f(l, mn.x, mn.y, mn.z);
        l = loc("u_inv_cell");     if (l >= 0) raytrace->uniform1f(l, 1.0f / cell_size);
        GLint gr2[3] = {grid_res.x, grid_res.y, grid_res.z};
        l = loc("u_grid_res");     if (l >= 0) raytrace->uniform3iv(l, gr2);
        l = loc("u_num_surfels");  if (l >= 0) raytrace->uniform1ui(l, GLuint(N));
        l = loc("u_light_pos");    if (l >= 0) raytrace->uniform3f(l, light_pos.x, light_pos.y, light_pos.z);
        l = loc("u_light_color");  if (l >= 0) raytrace->uniform3f(l, light_color.x, light_color.y, light_color.z);
        l = loc("u_light_intensity"); if (l >= 0) raytrace->uniform1f(l, light_intensity);
        l = loc("u_max_bounces");  if (l >= 0) raytrace->uniform1i(l, max_bounces);
        l = loc("u_search_radius"); if (l >= 0) raytrace->uniform1f(l, search_radius);
        l = loc("u_radius_scale"); if (l >= 0) raytrace->uniform1f(l, radius_scale);
        l = loc("u_ambient");      if (l >= 0) raytrace->uniform1f(l, ambient);
        int aux_sel = (view_mode >= 1) ? (view_mode - 1) : 0;  // albedo/normal/pos/depth
        l = loc("u_aux_mode");     if (l >= 0) raytrace->uniform1i(l, aux_sel);

        gl::dispatch_compute((fw + 7) / 8, (fh + 7) / 8, 1);
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        t_ray.end();
        traced = true;
    };

    while (!window.should_close()) {
        double now = window.time();
        float dt = float(now - last);
        last = now;
        ++frame_counter;

        window.poll_events();
        camera_control(window, cam, dt, !gui.wants_mouse(), captured);
        cam.set_aspect(float(window.framebuffer_width()) / float(window.framebuffer_height()));

        int fw = window.framebuffer_width(), fh = window.framebuffer_height();
        static int prev_fw = -1, prev_fh = -1;
        if (fw != prev_fw || fh != prev_fh) {
            create_rt_tex(fw, fh);
            create_gbuffer(gbuf_t, fw, fh);
            prev_fw = fw;
            prev_fh = fh;
        }

        // Hot-reload polls.
        if (display_prog.poll()) display = display_prog.take_program();
        if (gbuf_prog.poll()) gbuf = gbuf_prog.take_program();
        if (raytrace_prog.poll()) raytrace = raytrace_prog.take_program();
        if (pc_prog.poll()) pc = pc_prog.take_program();
        if (box_prog.poll()) boxp = box_prog.take_program();
        if (count_prog.poll()) count_cs = count_prog.take_program();
        if (scan_block_prog.poll()) scan_block_cs = scan_block_prog.take_program();
        if (scan_blocks_prog.poll()) scan_blocks_cs = scan_blocks_prog.take_program();
        if (scan_finish_prog.poll()) scan_finish_cs = scan_finish_prog.take_program();
        if (compact_prog.poll()) compact_cs = compact_prog.take_program();

        // Rasterize the direct view every frame (cheap).
        run_gbuffer();

        // Composite (mirror reflections): on demand or realtime.
        if (realtime) run_raytrace();

        // Display pass.
        t_display.begin();
        gl::disable(GL_DEPTH_TEST);
        gl::viewport(0, 0, fw, fh);
        gl::clear_color(0.02f, 0.02f, 0.03f, 1.0f);
        gl::clear(GL_COLOR_BUFFER_BIT);

        if (display->valid()) {
            display->use();
            auto dl = [&](const char* n) { return display->uniform_location(n); };
            GLint l;
            // view 0 = composite (reflections) when traced, else rasterized direct.
            // debug views = composite aux when traced, else rasterized G-buffer.
            gl::Texture* src = (view_mode == 0)
                ? (traced ? &color_tex : &gbuf_t.direct)
                : (traced ? &aux_tex
                          : (view_mode == 1 ? &gbuf_t.albedo
                                            : (view_mode == 2 ? &gbuf_t.normal
                                                              : &gbuf_t.position)));
            src->bind(0);
            l = dl("u_tex");      if (l >= 0) display->uniform1i(l, 0);
            l = dl("u_exposure"); if (l >= 0) display->uniform1f(l, exposure);
            l = dl("u_gamma");    if (l >= 0) display->uniform1f(l, gamma);
            fsq_vao.bind();
            gl::draw_arrays(GL_TRIANGLES, 0, 3);
        }
        t_display.end();

        // Headless test hook: screenshot + exit.
        if (autotrace && frame_counter == uint64_t(shot_frame)) {
            gfx::screenshot(shot_path ? shot_path : "rt38.png");
            gllib::logf(gllib::LogLevel::info, "saved %s, ray=%.3f ms",
                        shot_path ? shot_path : "rt38.png", t_ray.readback());
            break;
        }

        // Point cloud overlay.
        if (show_points && pc->valid()) {
            pc->use();
            auto pl = [&](const char* n) { return pc->uniform_location(n); };
            GLint l;
            glm::mat4 vp = cam.view_projection();
            l = pl("u_view_proj");  if (l >= 0) pc->uniform_matrix4fv(l, glm::value_ptr(vp));
            l = pl("u_point_size"); if (l >= 0) pc->uniform1f(l, point_size);
            l = pl("u_color_mode"); if (l >= 0) pc->uniform1i(l, pc_color_mode);
            l = pl("u_num_surfels"); if (l >= 0) pc->uniform1ui(l, GLuint(N));
            gl::enable(GL_PROGRAM_POINT_SIZE);
            gl::clear(GL_DEPTH_BUFFER_BIT);
            gl::enable(GL_DEPTH_TEST);
            gl::depth_func(GL_LESS);
            pc_vao.bind();
            gl::draw_arrays(GL_POINTS, 0, N);
            gl::disable(GL_PROGRAM_POINT_SIZE);
        }

        // Occupied-cell wireframe overlay.
        if (show_cells && boxp->valid() && !occ_aabbs.empty()) {
            boxp->use();
            auto bl = [&](const char* n) { return boxp->uniform_location(n); };
            GLint l;
            glm::mat4 vp = cam.view_projection();
            l = bl("u_view_proj"); if (l >= 0) boxp->uniform_matrix4fv(l, glm::value_ptr(vp));
            l = bl("u_color");     if (l >= 0) boxp->uniform4f(l, 0.0f, 1.0f, 0.0f, 0.6f);
            gl::enable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            gl::enable(GL_DEPTH_TEST);
            gl::depth_func(GL_LESS);
            cube_mesh.vao.bind();
            glDrawElementsInstanced(GL_TRIANGLES, cube_mesh.index_count,
                                    GL_UNSIGNED_INT, nullptr, GLsizei(occ_aabbs.size()));
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            gl::disable(GL_BLEND);
        }

        // ImGui.
        gui.begin_frame();
        {
            ImGui::Begin("38 Surfel Ray Tracing");
            ImGui::Text("FPS: %.1f   Frame: %.2f ms", 1.0f / std::max(dt, 1e-6f), dt * 1000.0f);
            ImGui::Text("Resolution: %d x %d", fw, fh);
            ImGui::Separator();

            if (ImGui::Button("Ray Trace")) run_raytrace();
            ImGui::SameLine();
            ImGui::Checkbox("Realtime", &realtime);
            ImGui::Text("Last ray trace: %.3f ms", t_ray.readback());
            ImGui::Text("Direct view: rasterized | Mirror: %s", traced ? "ray traced" : "not traced yet");
            ImGui::TextWrapped("Orbit: RMB drag  |  Zoom: scroll  |  Move: WASD");

            ImGui::Separator();
            ImGui::Combo("View", &view_mode, "Raytraced\0Albedo\0Normal\0Position\0Depth\0");
            ImGui::SliderFloat("Exposure", &exposure, 0.05f, 5.0f);
            ImGui::SliderFloat("Gamma", &gamma, 1.0f, 3.0f);

            ImGui::Separator();
            ImGui::Text("Surfels (N=%d)", N);
            static int last_budget_index = point_budget_index;
            ImGui::SliderInt("Point budget (index)", &point_budget_index, 0, kBudgetCount - 1, "%d");
            if (point_budget_index != last_budget_index) {
                last_budget_index = point_budget_index;
                build_surfels();
                search_radius = 0.0f;
                for (const auto& s : surfels) search_radius = std::max(search_radius, s.radius);
                compute_grid_params(2.0f * search_radius);
                cell_count_buf.data(nullptr, size_t(grid_volume) * sizeof(uint32_t));
                cell_start_buf.data(nullptr, size_t(grid_volume) * sizeof(uint32_t));
                block_sum_buf.data(nullptr, kScanBlocksMax * sizeof(uint32_t));
                std::vector<glm::vec4> vp2(N), vn2(N), va2(N);
                for (int i = 0; i < N; ++i) {
                    vp2[i] = glm::vec4(surfels[i].pos, surfels[i].radius);
                    vn2[i] = glm::vec4(surfels[i].nrm, 0.0f);
                    va2[i] = glm::vec4(surfels[i].alb, surfels[i].is_mirror ? 1.0f : 0.0f);
                }
                surfel_pos_buf.data(vp2.data(), vp2.size() * sizeof(glm::vec4));
                surfel_nrm_buf.data(vn2.data(), vn2.size() * sizeof(glm::vec4));
                surfel_alb_buf.data(va2.data(), va2.size() * sizeof(glm::vec4));
                std::vector<glm::vec4> cuts2 = compute_surfel_cuts(surfels, search_radius);
                surfel_cut_buf.data(cuts2.data(), cuts2.size() * sizeof(glm::vec4));
                surfel_slot_buf.data(nullptr, size_t(N) * sizeof(uint32_t));
                packed_idx_buf.data(nullptr, size_t(N) * sizeof(uint32_t));
                build_grid();
                refresh_occupied_cells();
                traced = false;
            }
            ImGui::Text("Leaf radius r = %.5f", leaf_radius);
            ImGui::Text("Cell size = %.4f (%dx%dx%d)", cell_size, grid_res.x, grid_res.y, grid_res.z);
            ImGui::Checkbox("Show point cloud", &show_points);
            if (show_points) {
                ImGui::Combo("Point color", &pc_color_mode, "Albedo\0Normal\0Position\0Mirror\0");
                ImGui::SliderFloat("Point size", &point_size, 1.0f, 20.0f);
            }
            ImGui::Checkbox("Show occupied cells", &show_cells);
            if (ImGui::Button("Rebuild grid")) {
                build_grid();
                refresh_occupied_cells();
            }
            ImGui::SameLine();
            ImGui::TextWrapped("(grid built on GPU via counting sort + prefix sum + compact)");

            ImGui::Separator();
            ImGui::Text("Light (point)");
            ImGui::SliderFloat("Intensity", &light_intensity, 0.0f, 50.0f);
            ImGui::SliderFloat("Ambient", &ambient, 0.0f, 1.0f);
            ImGui::SliderFloat("L x", &light_pos.x, mn.x, mx.x);
            ImGui::SliderFloat("L y", &light_pos.y, mn.y, mx.y);
            ImGui::SliderFloat("L z", &light_pos.z, mn.z, mx.z);
            ImGui::ColorEdit3("L color", &light_color.x);

            ImGui::Separator();
            ImGui::SliderInt("Mirror bounces", &max_bounces, 0, 3);
            ImGui::SliderFloat("Cylinder radius scale", &radius_scale, 0.5f, 2.0f);

            ImGui::Separator();
            ImGui::Text("GPU timings (ms)");
            ImGui::Text("  raytrace  %6.2f", t_ray.readback());
            ImGui::Text("  display   %6.2f", t_display.readback());
            ImGui::TextWrapped("Technique: Schaufler & Jensen 2000, ray tracing point-sampled geometry "
                               "through a GPU sparse uniform grid (DDA + interpolated surfel disks).");
            ImGui::End();
        }
        gui.render();

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
// Example 38 — Surfel Ray Tracing (Schaufler & Jensen 2000)
//
// "Ray Tracing Point Sampled Geometry" — the scene is represented as a surfel
// cloud (points with position/normal/albedo/radius). Sampling density is a
// user-facing step size and adapts per triangle to smooth-edge curvature:
// flat walls stay coarse, curved detail (Stanford bunny) refines to
// millimeter spacing, and each surfel stores its own disk radius. The cloud
// is accelerated by a multi-LOD set of sparse uniform grids built entirely on
// the GPU (per level: counting sort + prefix sum + compact; level 0 is
// coarsest and spans the scene, finer levels bound only their own surfels'
// bbox; a surfel's level follows its radius so cell >= 2r holds per level).
// Rays DDA through the coarsest grid and descend into finer levels clipped
// to each coarse cell's t-interval; the first surfel disk hit is found per
// the paper, and the surface is reconstructed by interpolating nearby
// surfels weighted by (r - d_i) (paper Eq. 1), gathered from all levels.
// Sharp mesh feature edges clip surfel disks exactly (per-surfel cut planes
// in anchor form). Shading is a single point light with 1/d^2 falloff (no
// shadows); mirror materials spawn perfect specular reflection rays.
//
// The same grids are the intended acceleration structure for a future micro-
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
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

constexpr float kPi      = 3.14159265358979f;
constexpr float kLATTICE_R = 0.8f;    // stored disk radius = this * local grid step h
constexpr float kSHARP_DOT = 0.6f;    // |n_i . n_j| below this => sharp edge
constexpr uint32_t kSEED = 42;
constexpr float kCurvAdapt    = 4.5f;   // curvature adaptivity gamma (see sampler)
constexpr float kCurvKappaMax = 200.0f; // curvature clamp [1/m]
constexpr float kCurvFloor    = 0.08f;  // min adaptive step, as fraction of h_base
constexpr int   kBlockSize = 256;
constexpr int   kScanBlocksMax = 1024;   // scan_blocks.comp single workgroup limit
constexpr int   kMaxGridVolume = kBlockSize * kScanBlocksMax;  // 262144
constexpr int   kGridLevels = 4;         // multi-LOD grid levels (cell halves per level)

// ---------------------------------------------------------------------------
// Triangle + surfel structures (CPU-side scene preparation)
// ---------------------------------------------------------------------------

struct Tri {
    glm::vec3 p[3];
    glm::vec3 n[3];
    float area = 0.0f;
    float curvature = 0.0f;   // max smooth-edge dihedral / edge length [1/m]
    int object = 0;           // connected-component id (see extract_triangles)
    glm::vec3 alb{0.0f};
};

struct SamplePoint {
    glm::vec3 pos{0.0f};
    glm::vec3 nrm{0.0f};
    glm::vec3 alb{0.0f};
    float radius = 0.0f;   // per-sample disk radius (adaptive step derived)
    bool is_mirror = false;
    int object = 0;        // home triangle's object id (per-object cuts)
};

// Sharp/boundary mesh feature edge (world space). Cut planes for the surfel
// disks are anchored to these, so clipped disks end exactly at triangle
// borders (paper §3: border gaps need special handling).
struct FeatEdge {
    glm::vec3 mid;       // midpoint of the edge
    glm::vec3 dir;       // unit edge direction
    float half_len = 0.0f;
    glm::vec3 na, nb;    // incident face normals (nb unused when boundary)
    glm::vec3 ctr_a, ctr_b;  // incident face centroids (ctr_b unused when boundary)
    glm::vec3 tri_a[3];  // incident triangle of face a (boundary ownership test)
    int obj_a, obj_b;    // incident faces' object ids (per-object cutting)
    bool boundary = false;
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
    std::vector<FeatEdge> edges;    // sharp + boundary feature edges
};

ExtractedMesh extract_triangles(const gfx::Model& model, const glm::mat4& model_mat) {
    ExtractedMesh out;

    for (size_t mi = 0; mi < model.mesh_count(); ++mi) {
        const gfx::Mesh& mesh = model.mesh(int(mi));
        int mati = model.mesh_material(int(mi));
        const auto& mat = model.material_info(size_t(mati >= 0 ? mati : 0));
        glm::vec3 alb(mat.base_color_factor[0], mat.base_color_factor[1], mat.base_color_factor[2]);
        bool is_mirror = (mat.name.find("tallBox") != std::string::npos) ||
                         (mat.roughness_factor < 0.05f);
        
        // Combine the mesh's node transform with the global model transform
        glm::mat4 combined_xform = model_mat * model.mesh_transform(int(mi));
        glm::mat3 normal_mat = glm::transpose(glm::inverse(glm::mat3(combined_xform)));
        
        const auto& vs = mesh.vertices();
        const auto& is = mesh.indices();

        auto add_tri = [&](const gfx::Vertex& a, const gfx::Vertex& b, const gfx::Vertex& c) {
            Tri t;
            glm::vec3 pa(a.position[0], a.position[1], a.position[2]);
            glm::vec3 pb(b.position[0], b.position[1], b.position[2]);
            glm::vec3 pc(c.position[0], c.position[1], c.position[2]);
            t.p[0] = glm::vec3(combined_xform * glm::vec4(pa, 1.0f));
            t.p[1] = glm::vec3(combined_xform * glm::vec4(pb, 1.0f));
            t.p[2] = glm::vec3(combined_xform * glm::vec4(pc, 1.0f));
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

    // Feature edges: hash triangle edges by quantized endpoint positions and
    // collect incident face normals (geometric, from baked winding). Edges with
    // one incident triangle are boundaries (open space beyond); edges whose two
    // faces differ sharply are creases. Coplanar diagonals drop out on their own.
    struct EdgeAccum {
        int count = 0;
        int ti[2];          // incident triangle indices
        glm::vec3 n[2];
        glm::vec3 ctr[2];   // centroids of the incident triangles
        glm::vec3 p[2][3];
        glm::vec3 opp[2];   // third vertex of each incident triangle
    };
    std::map<std::pair<int64_t,int64_t>, EdgeAccum> edge_map;
    auto qkey = [&](const glm::vec3& p) -> int64_t {
        return int64_t(std::llround(p.x * 10000.0f)) * 73856093 ^
               int64_t(std::llround(p.y * 10000.0f)) * 19349663 ^
               int64_t(std::llround(p.z * 10000.0f)) * 83492791;
    };
    for (size_t ti = 0; ti < out.tris.size(); ++ti) {
        const Tri& t = out.tris[ti];
        glm::vec3 fn = glm::normalize(glm::cross(t.p[1] - t.p[0], t.p[2] - t.p[0]));
        glm::vec3 ctr = (t.p[0] + t.p[1] + t.p[2]) / 3.0f;
        for (int k = 0; k < 3; ++k) {
            const glm::vec3& p0 = t.p[k];
            const glm::vec3& p1 = t.p[(k + 1) % 3];
            auto ka = qkey(p0), kb = qkey(p1);
            if (ka > kb) std::swap(ka, kb);
            EdgeAccum& e = edge_map[{ka, kb}];
            if (e.count < 2) {
                e.ti[e.count] = int(ti);
                e.n[e.count] = fn;
                e.ctr[e.count] = ctr;
                e.p[e.count][0] = p0;
                e.p[e.count][1] = p1;
                e.opp[e.count] = t.p[0] + t.p[1] + t.p[2] - p0 - p1;
            }
            ++e.count;
        }
    }
    // Per-triangle curvature from SMOOTH edge dihedrals: kappa = angle / length.
    // Drives adaptive sampling density (flat walls stay coarse, curved detail
    // refines). Geometric face normals only — immune to bad vertex normals.
    // Sharp creases are excluded: the two faces are flat; edge fidelity there
    // comes from the cut planes, not from sample density. On very short edges
    // kappa ~ dihedral/len grows, but the sample demand of a triangle is
    // ~area*kappa² which stays bounded (dihedral-dominated), so no explosion.
    //
    // Object labeling: triangles joined by a shared edge belong to the same
    // object (union-find). Separate meshes that merely touch or interpenetrate
    // (the bunny sunk into the box top) stay SEPARATE objects, so feature
    // edges of one object never cut another object's surfels.
    {
        std::vector<int> parent(out.tris.size());
        for (size_t i = 0; i < parent.size(); ++i) parent[i] = int(i);
        auto find = [&](int x) {
            while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
            return x;
        };
        for (auto& [key, e] : edge_map) {
            if (e.count < 2) continue;
            int a = find(e.ti[0]), b = find(e.ti[1]);
            if (a != b) parent[a] = b;
        }
        std::unordered_map<int, int> comp;
        int next_comp = 0;
        for (size_t i = 0; i < out.tris.size(); ++i) {
            const int r = find(int(i));
            auto it = comp.find(r);
            if (it == comp.end()) it = comp.emplace(r, next_comp++).first;
            out.tris[i].object = it->second;
        }
    }
    for (auto& [key, e] : edge_map) {
        if (e.count != 2) continue;
        const float dn = std::abs(glm::dot(e.n[0], e.n[1]));
        if (dn < kSHARP_DOT) continue;   // sharp crease: not curvature
        const float ang = std::acos(std::clamp(dn, 0.0f, 1.0f));
        const float len = std::max(glm::length(e.p[0][1] - e.p[0][0]), 1e-6f);
        const float kappa = std::min(ang / len, kCurvKappaMax);
        out.tris[size_t(e.ti[0])].curvature = std::max(out.tris[size_t(e.ti[0])].curvature, kappa);
        out.tris[size_t(e.ti[1])].curvature = std::max(out.tris[size_t(e.ti[1])].curvature, kappa);
    }
    out.edges.reserve(edge_map.size());
    for (auto& [key, e] : edge_map) {
        FeatEdge fe;
        fe.mid = 0.5f * (e.p[0][0] + e.p[0][1]);
        fe.dir = glm::normalize(e.p[0][1] - e.p[0][0]);
        fe.half_len = 0.5f * glm::length(e.p[0][1] - e.p[0][0]);
        if (e.count == 1) {
            fe.boundary = true;
            fe.na = e.n[0];
            fe.ctr_a = e.ctr[0];
            fe.tri_a[0] = e.p[0][0];
            fe.tri_a[1] = e.p[0][1];
            fe.tri_a[2] = e.opp[0];
            fe.obj_a = out.tris[size_t(e.ti[0])].object;
            fe.obj_b = fe.obj_a;
            out.edges.push_back(fe);
        } else if (e.count == 2 && std::abs(glm::dot(e.n[0], e.n[1])) < kSHARP_DOT) {
            fe.boundary = false;
            fe.na = e.n[0];
            fe.nb = e.n[1];
            fe.ctr_a = e.ctr[0];
            fe.ctr_b = e.ctr[1];
            fe.obj_a = out.tris[size_t(e.ti[0])].object;
            fe.obj_b = out.tris[size_t(e.ti[1])].object;
            out.edges.push_back(fe);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Barycentric lattice sampling with curvature-adaptive density.
//
// Instead of random best-candidate placement, each triangle carries a regular
// lattice aligned to its edges: p0 + (u/n1) e1 + (v/n2) e2 for u,v >= 0 inside
// u/n1 + v/n2 <= 1. All three triangle borders receive rows lying exactly on
// them, spacing is uniform, and the biggest hole is known analytically (paper
// §3/§4.1 require r > biggest hole): lattice covering radius is at most
// sqrt(h^2+h^2)/2 <= h*sqrt(2)/2, so a disk radius kLATTICE_R * h with
// kLATTICE_R > 0.71 makes the point-sampled surface provably watertight.
//
// The lattice step is PER TRIANGLE: h_i = h_base / (1 + gamma * kappa_i *
// h_base), where kappa_i is the triangle's smooth-edge curvature. Flat surfaces
// (walls) sample at h_base; curved detail (bunny: kappa ~ 15-50 1/m) refines
// down to millimeter spacing. Each sample stores its own disk radius
// kLATTICE_R * h_i — the whole pipeline (grid levels, traversal, blending)
// already consumes per-surfel radii. Sub-step triangles (edges shorter than
// h_i — dense imported meshes) place samples proportional to AREA via error
// diffusion so their population converges to ~area/h_i²; one sample lands at
// the centroid, extras spread deterministically (golden-ratio barycentrics).
// The sample count is an OUTPUT of the density — there is no budget rescale.
// ---------------------------------------------------------------------------

static std::vector<SamplePoint> sample_points_lattice(
    const std::vector<Tri>& tris, const std::vector<bool>& tri_mirror,
    float h_base, float& step_out)
{
    auto generate = [&](float h) {
        std::vector<SamplePoint> pts;
        float carry = 0.0f;
        for (size_t i = 0; i < tris.size(); ++i) {
            const Tri& t = tris[i];
            glm::vec3 e1 = t.p[1] - t.p[0];
            glm::vec3 e2 = t.p[2] - t.p[0];
            // Curvature-adaptive local step (floored at a fraction of h_base
            // so degenerate kappa estimates cannot explode the count).
            const float hi = std::max(h / (1.0f + kCurvAdapt * std::min(t.curvature, kCurvKappaMax) * h),
                                      h * kCurvFloor);
            const float ri = kLATTICE_R * hi;
            int n1 = std::max(1, int(std::round(glm::length(e1) / hi)));
            int n2 = std::max(1, int(std::round(glm::length(e2) / hi)));
            if (n1 == 1 && n2 == 1) {
                // Sub-step triangle (edges shorter than the local lattice step).
                // Corner sampling would emit 3 samples per triangle regardless
                // of step (count explosion) and pile duplicates onto shared
                // vertices. Place samples PROPORTIONAL TO AREA via error
                // diffusion so the population converges to ~area/h_i².
                const float target = t.area / (hi * hi);
                const int n = int(target + carry);
                carry += target - float(n);
                for (int j = 0; j < n; ++j) {
                    SamplePoint s;
                    if (j == 0) {
                        s.pos = (t.p[0] + t.p[1] + t.p[2]) / 3.0f;
                    } else {
                        float a = std::fmod(float(j) * 0.618034f + 0.5f, 1.0f);
                        float b = std::fmod(float(j) * 0.754878f + 0.25f, 1.0f);
                        if (a + b > 1.0f) { a = 1.0f - a; b = 1.0f - b; }
                        s.pos = t.p[0] + e1 * a + e2 * b;
                    }
                    s.nrm = glm::normalize(t.n[0] + t.n[1] + t.n[2]);
                    s.alb = t.alb;
                    s.is_mirror = tri_mirror[i];
                    s.object = t.object;
                    s.radius = ri;
                    pts.push_back(s);
                }
                continue;
            }
            for (int u = 0; u <= n1; ++u)
                for (int v = 0; v <= n2; ++v) {
                    if (float(u) / float(n1) + float(v) / float(n2) > 1.0f + 1e-6f)
                        continue;   // keep barycentric w >= 0 half-parallelogram
                    SamplePoint s;
                    s.pos = t.p[0] + e1 * (float(u) / float(n1))
                          + e2 * (float(v) / float(n2));
                    s.nrm = glm::normalize(t.n[0] + t.n[1] + t.n[2]);
                    s.alb = t.alb;
                    s.is_mirror = tri_mirror[i];
                    s.object = t.object;
                    s.radius = ri;
                    pts.push_back(s);
                }
        }
        return pts;
    };

    std::vector<SamplePoint> pts = generate(h_base);
    step_out = h_base;
    return pts;
}

// ---------------------------------------------------------------------------
// Surfel edge cutting: for each surfel, store up to K cut planes so the shader
// can clip its disk exactly at mesh feature edges (paper Sec. 3: border gaps
// need special treatment).
//
// Cuts are derived from topology (extracted sharp/boundary edges), not from
// neighbor samples: planes land exactly on triangle borders regardless of local
// sampling. Which side of an edge a surfel's disk lives on is decided from the
// INCIDENT FACE GEOMETRY (centroids), never from the surfel center or the sign
// of a computed offset: the barycentric lattice places whole rows of samples
// exactly ON the triangle borders, so those quantities are exactly zero there
// and sign-based decisions flip cuts the wrong way round for roughly half of
// the border samples (disks keeping the outside half-plane => overextension;
// whole disks rejected at concave seams => hairline gaps). Cuts are EXACT at
// every edge: both faces' border rows end at the seam line, so the clipped
// disk sets meet watertight — no overlap poke is needed, and attributes do not
// blend across the crease. Shared edges only cut surfels of their two incident
// faces, boundary edges only surfels lying on the incident triangle (other
// surfaces merely pass close by — a box standing on the floor must not be
// clipped by the floor, or vice versa).
// Shader clip condition (anchor form): reject p if dot(p - anchor, n) > 0.
// Each cut is stored as TWO vec4s: [n.xyz, 0] and [anchor.xyz, 0]. Empty slots
// have n = 0 (dot == 0, never rejected). The anchor form keeps the layout
// ready for per-frame skinned cuts (skin n like a normal, anchor like a point).
// ---------------------------------------------------------------------------

constexpr int kSurfelCuts = 6;

std::vector<glm::vec4> compute_surfel_cuts(const std::vector<SamplePoint>& pts,
                                           const std::vector<FeatEdge>& edges,
                                           float search_radius, float ref_radius) {
    (void)search_radius;
    (void)ref_radius;
    const int N = int(pts.size());
    std::vector<glm::vec4> cuts(size_t(N) * kSurfelCuts * 2, glm::vec4(0.0f));

    for (int i = 0; i < N; ++i) {
        const glm::vec3& ca = pts[i].pos;
        // Per-surfel reach: the disk can only cross an edge within its own
        // effective extent (radius * max radius_scale of 2). The old global
        // 2*search_radius reach propagated crevice cuts across whole curved
        // surfaces — on the bunny, chin/neck contact edges beheaded surfels
        // 4cm away (a plane through a distant crease slices unrelated parts
        // of a curved surface; on flat walls it was harmless instead).
        const float reach = 2.0f * pts[i].radius;
        int ncuts = 0;
        // n: plane normal, keep dot(p - anchor, n) <= 0.
        auto add_cut = [&](const glm::vec3& n, const glm::vec3& anchor) {
            // A cut must never reject its own surfel: if the plane slices
            // through the surfel center, it is not a clean crease for this
            // disk but a shallow surface intersection (curved meshes like the
            // bunny self-intersect where legs merge into the body; the "other
            // face" plane is nearly tangent there and beheads the disk).
            // Border-row samples sit exactly ON their cut planes (|dot| ~ 1e-6)
            // and are unaffected by the tolerance.
            if (glm::dot(ca - anchor, n) > 1e-4f) return;
            for (int k = 0; k < ncuts; ++k) {
                const glm::vec4 ex_n = cuts[(size_t(i) * kSurfelCuts + k) * 2];
                const glm::vec4 ex_a = cuts[(size_t(i) * kSurfelCuts + k) * 2 + 1];
                if (glm::dot(glm::vec3(ex_n), n) > 0.98f &&
                    std::abs(glm::dot(glm::vec3(ex_a) - anchor, n)) < 1e-3f)
                    return;
            }
            if (ncuts >= kSurfelCuts) return;
            cuts[(size_t(i) * kSurfelCuts + ncuts) * 2]     = glm::vec4(n, 0.0f);
            cuts[(size_t(i) * kSurfelCuts + ncuts) * 2 + 1] = glm::vec4(anchor, 0.0f);
            ++ncuts;
        };
        for (const FeatEdge& e : edges) {
            if (ncuts >= kSurfelCuts) break;
            // Point-segment distance from the surfel center to the edge.
            float t = glm::clamp(glm::dot(ca - e.mid, e.dir), -e.half_len, e.half_len);
            glm::vec3 cp = e.mid + e.dir * t;
            if (glm::length(ca - cp) > reach) continue;
            if (!e.boundary) {
                // Shared sharp edge: only the two incident faces are clipped.
                // PER-OBJECT: an edge never cuts surfels of another object —
                // separate meshes that merely touch or interpenetrate (the
                // bunny sunk into the box top) would otherwise receive each
                // other's crevice cuts and punch holes into each other.
                if (e.obj_a != pts[i].object && e.obj_b != pts[i].object) continue;
                // Own face is identified by normal agreement; with flat face
                // normals this is exact (1 vs ~0).
                const float da = glm::dot(pts[i].nrm, e.na);
                const float db = glm::dot(pts[i].nrm, e.nb);
                if (glm::max(da, db) < 0.5f) continue;   // surfel of another surface
                const bool own_is_a = da >= db;
                const glm::vec3& other = own_is_a ? e.nb : e.na;
                const glm::vec3& own_ctr = own_is_a ? e.ctr_a : e.ctr_b;
                // Concavity from the OWN face's centroid: strictly inside the
                // face, so the test never degenerates (the surfel center can
                // sit exactly on the edge). Concave => own face extends into
                // the other face's half-space (crevice).
                const bool concave = glm::dot(other, own_ctr - cp) > 0.0f;
                // Plane through cp (ON the edge) with the other face's normal:
                // keep dot(p - cp, other) <= 0, i.e. reject the half-space on
                // the other face's side. The plane always passes exactly
                // through the edge — no offset math, so the float-noise flips
                // that plagued border-row samples (which sit exactly ON the
                // edge) cannot occur. Concave faces keep their own side: flip
                // the normal, same anchor.
                add_cut(concave ? -other : other, cp);
            } else {
                // Boundary edge: clip only surfels lying ON the incident
                // triangle of the SAME object — other surfaces passing within
                // reach (a box standing on the floor) keep their disks.
                if (e.obj_a != pts[i].object) continue;
                // Ownership test: the sample must sit on the triangle's plane
                // AND inside it (with tolerance; lattice samples lie exactly
                // on it).
                const float dp = glm::dot(ca - e.tri_a[0], e.na);
                if (std::abs(dp) > 1e-3f) continue;
                const glm::vec3 v0 = e.tri_a[1] - e.tri_a[0];
                const glm::vec3 v1 = e.tri_a[2] - e.tri_a[0];
                const glm::vec3 vq = ca - e.na * dp - e.tri_a[0];
                const float d00 = glm::dot(v0, v0), d01 = glm::dot(v0, v1);
                const float d11 = glm::dot(v1, v1);
                const float d20 = glm::dot(vq, v0), d21 = glm::dot(vq, v1);
                const float den = d00 * d11 - d01 * d01;
                if (std::abs(den) < 1e-12f) continue;
                const float bu = (d11 * d20 - d01 * d21) / den;
                const float bv = (d00 * d21 - d01 * d20) / den;
                const float bw = 1.0f - bu - bv;
                if (bu < -1e-3f || bv < -1e-3f || bw < -1e-3f) continue;
                // u points from the edge into the face interior, decided
                // against the incident face's centroid (strictly inside, so
                // never degenerate for border-row samples). Anchor form:
                // keep dot(p - cp, -u) <= 0 rejects exactly dot(p-cp,u) < 0.
                glm::vec3 u = glm::cross(e.dir, e.na);
                float ul = glm::length(u);
                if (ul < 1e-8f) continue;
                u /= ul;
                if (glm::dot(u, e.ctr_a - cp) < 0.0f) u = -u;   // edge -> interior
                add_cut(-u, cp);
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
constexpr uint32_t kCacheVersion = 8;   // 8: per-object cuts (object id per sample)

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
    uint32_t step_bits = 0;   // bit-cast base step (density factor)
    uint32_t seed = kSEED;
};

std::string cache_path_for(const std::string& model_path, float base_step) {
    std::filesystem::path p(model_path);
    return "cache/38_" + p.stem().string() + "_S" +
           std::to_string(int(std::round(base_step * 10000.0f))) + ".bin";
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
    w32(key.step_bits);
    w32(key.seed);
    wf(leaf_radius);
    wf(total_area);
    w32(uint32_t(pts.size()));   // actual sample count (may differ from key.N)
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
    if (r32() != key.step_bits) return false;
    if (r32() != key.seed) return false;

    leaf_radius = rf();
    total_area = rf();
    const uint32_t count = r32();
    if (count == 0 || count > 10u * 1000u * 1000u) return false;
    pts.resize(count);
    f.read(reinterpret_cast<char*>(pts.data()), count * sizeof(SamplePoint));
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

    // Model list.
    const char* model_paths[] = {
        "CornellBoxMirrorBox.glb",
        "CornellBoxBunnyMirror.glb",
        "CornellBoxMirrorBunny.glb",
    };
    const char* model_names[] = {
        "Cornell Box (Mirror)",
        "Cornell Box (Mirror) + Bunny",
        "Cornell Box + Bunny (Mirror)",
    };
    constexpr int num_models = 3;
    int current_model_index = 0;
    if (const char* m = getenv("SRT_MODEL")) current_model_index = std::clamp(std::atoi(m), 0, num_models - 1);

    // Model.
    const char* model_path = model_paths[current_model_index];
    gfx::Model model;
    if (!model.load(model_path)) {
        gllib::logf(gllib::LogLevel::error, "Failed to load %s", model_path);
        return EXIT_FAILURE;
    }
    
    // The glTF models already have correct orientations encoded in their node transforms.
    // No additional global transform is needed.
    auto get_model_transform = [](int model_idx) -> glm::mat4 {
        (void)model_idx;  // All models use identity transform
        return glm::mat4(1.0f);
    };
    
    glm::mat4 model_mat = get_model_transform(current_model_index);

    gllib::logf(gllib::LogLevel::info, "Loaded: %zu meshes, %zu materials",
                model.mesh_count(), model.material_count());
    for (size_t i = 0; i < model.material_count(); ++i) {
        const auto& m = model.material_info(i);
        gllib::logf(gllib::LogLevel::info, "mat %zu '%s': albedo(%.3f %.3f %.3f) rough=%.3f",
                    i, m.name.c_str(), m.base_color_factor[0], m.base_color_factor[1],
                    m.base_color_factor[2], m.roughness_factor);
    }

    // ---- Surfel cloud (CPU, cached) ----
    float base_step = 0.05f;      // base lattice step (density factor) in meters
    if (getenv("SRT_STEP")) base_step = std::clamp(std::atof(getenv("SRT_STEP")), 0.002, 1.0);
    int N = 0;
    float leaf_radius = 0.0f;     // coarse reference disk radius (kLATTICE_R * base_step)
    float search_radius = 0.0f;   // max per-surfel radius, for grid/traversal bounds
    float total_area = 0.0f;

    // Load the initial model
    ExtractedMesh em;
    if (!model.load(model_path)) {
        gllib::logf(gllib::LogLevel::error, "Failed to load %s", model_path);
        return EXIT_FAILURE;
    }
    em = extract_triangles(model, model_mat);
    total_area = 0.0f;
    for (const auto& t : em.tris) total_area += t.area;

    gllib::logf(gllib::LogLevel::info, "Loaded: %zu meshes, %zu materials",
                model.mesh_count(), model.material_count());
    for (size_t i = 0; i < model.material_count(); ++i) {
        const auto& m = model.material_info(i);
        gllib::logf(gllib::LogLevel::info, "mat %zu '%s': albedo(%.3f %.3f %.3f) rough=%.3f",
                    i, m.name.c_str(), m.base_color_factor[0], m.base_color_factor[1],
                    m.base_color_factor[2], m.roughness_factor);
    }

    // Function to load a model and extract its triangles
    auto load_model_and_extract = [&](const char* new_model_path) -> bool {
        gfx::Model new_model;
        if (!new_model.load(new_model_path)) {
            gllib::logf(gllib::LogLevel::error, "Failed to load %s", new_model_path);
            return false;
        }
        model = std::move(new_model);
        model_path = new_model_path;
        
        gllib::logf(gllib::LogLevel::info, "Loaded: %zu meshes, %zu materials",
                    model.mesh_count(), model.material_count());
        for (size_t i = 0; i < model.material_count(); ++i) {
            const auto& m = model.material_info(i);
            gllib::logf(gllib::LogLevel::info, "mat %zu '%s': albedo(%.3f %.3f %.3f) rough=%.3f",
                        i, m.name.c_str(), m.base_color_factor[0], m.base_color_factor[1],
                        m.base_color_factor[2], m.roughness_factor);
        }
        
        // Extract triangles with the model transform
        em = extract_triangles(model, model_mat);
        total_area = 0.0f;
        for (const auto& t : em.tris) total_area += t.area;
        
        return true;
    };

    std::vector<SamplePoint> surfels;

    auto build_surfels = [&]() {
        CacheKey key;
        key.model_stamp = file_stamp(model_path, nullptr);
        key.step_bits = std::bit_cast<uint32_t>(base_step);
        key.mat_hash = material_hash(model);
        key.transform_hash = matrix_hash(model_mat);

        std::string cpath = cache_path_for(model_path, base_step);
        if (load_cache(cpath, key, surfels, leaf_radius, total_area)) {
            gllib::logf(gllib::LogLevel::info, "cache hit %s (step %.4f)", cpath.c_str(), base_step);
            leaf_radius = kLATTICE_R * base_step;
        } else {
            auto t0 = std::chrono::steady_clock::now();
            float step = 0.0f;
            surfels = sample_points_lattice(em.tris, em.tri_mirror, base_step, step);
            leaf_radius = kLATTICE_R * step;
            if (save_cache(cpath, key, surfels, leaf_radius, total_area))
                gllib::logf(gllib::LogLevel::info, "cached to %s", cpath.c_str());
            auto t1 = std::chrono::steady_clock::now();
            gllib::logf(gllib::LogLevel::info,
                        "sampled %zu surfels (step %.5f) in %.0f ms (%.1f tris)",
                        surfels.size(), step,
                        std::chrono::duration<double, std::milli>(t1 - t0).count(),
                        double(em.tris.size()));
        }
        // The sample count is an output of the density factor. N is the ACTUAL
        // count so buffers, dispatches and the grid cover the whole cloud.
        N = int(surfels.size());
    };
    build_surfels();

    // Feature edges are needed on every launch (cut planes are computed from
    // them even when the surfel cloud itself comes from the cache).
    
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
    // Multi-LOD grids: one combined uint buffer per array, segmented per level.
    gl::Buffer cell_count_all(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer cell_start_all(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer packed_all(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer level_idx_all(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer surfel_slot_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    gl::Buffer block_sum_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);

    std::vector<glm::vec4> vp, vn, va, cuts;
    auto upload_surfel_buffers = [&]() {
        vp.assign(size_t(N), glm::vec4(0.0f));
        vn.assign(size_t(N), glm::vec4(0.0f));
        va.assign(size_t(N), glm::vec4(0.0f));
        for (int i = 0; i < N; ++i) {
            vp[size_t(i)] = glm::vec4(surfels[i].pos, surfels[i].radius);
            vn[size_t(i)] = glm::vec4(surfels[i].nrm, 0.0f);
            va[size_t(i)] = glm::vec4(surfels[i].alb, surfels[i].is_mirror ? 1.0f : 0.0f);
        }
        surfel_pos_buf.data(vp.data(), vp.size() * sizeof(glm::vec4));
        surfel_nrm_buf.data(vn.data(), vn.size() * sizeof(glm::vec4));
        surfel_alb_buf.data(va.data(), va.size() * sizeof(glm::vec4));
        cuts = compute_surfel_cuts(surfels, em.edges, search_radius, leaf_radius);
        surfel_cut_buf.data(cuts.data(), cuts.size() * sizeof(glm::vec4));
    };
    upload_surfel_buffers();

    // ---- Multi-LOD sparse grids ----
    // Level 0 is the coarsest and spans the whole scene; finer levels cover only
    // the bbox of their (smaller-radius) surfels. A surfel's level is the
    // smallest l with cell_l >= 2*r_s, cell_l = c0 / 2^l, c0 = 2*search_radius,
    // which preserves the traversal invariant "any disk intersecting the ray
    // cylinder lies within the 3x3x3 neighborhood of the ray's cell AT ITS OWN
    // LEVEL" — the same guarantee the single grid had, per level.
    struct LevelSpec {
        glm::vec3 mn{0.0f};
        float cell = 1.0f;
        glm::ivec3 res{1};
        uint32_t volume = 0;
        uint32_t count = 0;                                    // surfels assigned
        uint32_t cnt_off = 0, start_off = 0, packed_off = 0, idx_off = 0;
    };
    LevelSpec levels[kGridLevels];
    uint32_t cnt_total = 0, start_total = 0, packed_total = 0, idx_total = 0;
    std::vector<uint32_t> level_of;

    auto compute_level_specs = [&]() {
        std::vector<glm::vec3> lmn(kGridLevels, glm::vec3(1e30f));
        std::vector<glm::vec3> lmx(kGridLevels, glm::vec3(-1e30f));
        std::vector<uint32_t> lcount(kGridLevels, 0u);
        level_of.assign(size_t(N), 0u);
        const float c0 = 2.0f * search_radius;
        for (int i = 0; i < N; ++i) {
            const float r = surfels[i].radius;
            int l = 0;
            if (r > 0.0f && c0 > 0.0f)
                l = int(std::floor(std::log2(c0 / (2.0f * r))));
            l = std::clamp(l, 0, kGridLevels - 1);
            level_of[size_t(i)] = uint32_t(l);
            lmn[size_t(l)] = glm::min(lmn[size_t(l)], surfels[i].pos);
            lmx[size_t(l)] = glm::max(lmx[size_t(l)], surfels[i].pos);
            ++lcount[size_t(l)];
        }
        cnt_total = start_total = packed_total = idx_total = 0;
        for (int l = 0; l < kGridLevels; ++l) {
            LevelSpec& L = levels[l];
            L.count = lcount[size_t(l)];
            L.cell = c0 / float(1 << l);
            if (L.count == 0) {
                L.mn = mn;
                L.res = glm::ivec3(1);
                L.volume = 1;
            } else {
                L.mn = lmn[size_t(l)] - glm::vec3(L.cell);   // 1-cell margin
                const glm::vec3 span = (lmx[size_t(l)] - lmn[size_t(l)]) + glm::vec3(2.0f * L.cell);
                for (int a = 0; a < 3; ++a)
                    L.res[a] = std::max(1, int(std::ceil(span[a] / L.cell)));
                float cs = L.cell;
                while (L.res.x * L.res.y * L.res.z > kMaxGridVolume) {
                    cs *= 2.0f;
                    for (int a = 0; a < 3; ++a)
                        L.res[a] = std::max(1, int(std::ceil(span[a] / cs)));
                }
                L.cell = cs;
                L.volume = uint32_t(L.res.x) * uint32_t(L.res.y) * uint32_t(L.res.z);
            }
            L.cnt_off = cnt_total;       cnt_total += L.volume;
            L.start_off = start_total;   start_total += L.volume;
            L.packed_off = packed_total; packed_total += L.count;
            L.idx_off = idx_total;       idx_total += L.count;
        }
    };

    auto upload_level_buffers = [&]() {
        compute_level_specs();
        std::vector<uint32_t> idx(idx_total, 0u);
        for (int l = 0; l < kGridLevels; ++l) {
            uint32_t w = levels[l].idx_off;
            for (int i = 0; i < N; ++i)
                if (level_of[size_t(i)] == uint32_t(l)) idx[w++] = uint32_t(i);
        }
        level_idx_all.data(idx.data(), idx.size() * sizeof(uint32_t));
        cell_count_all.data(nullptr, size_t(cnt_total) * sizeof(uint32_t));
        cell_start_all.data(nullptr, size_t(start_total) * sizeof(uint32_t));
        packed_all.data(nullptr, size_t(packed_total) * sizeof(uint32_t));
        surfel_slot_buf.data(nullptr, size_t(N) * sizeof(uint32_t));
        block_sum_buf.data(nullptr, kScanBlocksMax * sizeof(uint32_t));
    };
    upload_level_buffers();

    const uint32_t zero = 0;

    // ---- Grid build (GPU): count -> scan -> compact, once per level ----
    auto set_level_uniforms = [&](gl::Program& prog, int l) {
        auto loc = [&](const char* n) { return prog.uniform_location(n); };
        GLint u;
        u = loc("u_grid_min");    if (u >= 0) prog.uniform3f(u, levels[l].mn.x, levels[l].mn.y, levels[l].mn.z);
        u = loc("u_inv_cell");    if (u >= 0) prog.uniform1f(u, 1.0f / levels[l].cell);
        GLint gr[3] = {levels[l].res.x, levels[l].res.y, levels[l].res.z};
        u = loc("u_grid_res");    if (u >= 0) prog.uniform3iv(u, gr);
        u = loc("u_num");         if (u >= 0) prog.uniform1ui(u, levels[l].count);
        u = loc("u_cnt_off");     if (u >= 0) prog.uniform1ui(u, levels[l].cnt_off);
        u = loc("u_start_off");   if (u >= 0) prog.uniform1ui(u, levels[l].start_off);
        u = loc("u_packed_off");  if (u >= 0) prog.uniform1ui(u, levels[l].packed_off);
        u = loc("u_idx_off");     if (u >= 0) prog.uniform1ui(u, levels[l].idx_off);
    };

    auto bind_rt = [&] {
        surfel_pos_buf.bind_base(0);
        surfel_nrm_buf.bind_base(1);
        surfel_alb_buf.bind_base(2);
        cell_start_all.bind_base(3);
        cell_count_all.bind_base(4);
        packed_all.bind_base(5);
        surfel_cut_buf.bind_base(6);
    };

    auto build_grid = [&]() {
        // Zero ALL level count segments at once.
        cell_count_all.clear(GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
        for (int l = 0; l < kGridLevels; ++l) {
            if (levels[l].count == 0) continue;
            const uint32_t n = levels[l].count;

            // Pass 1: count + slot.
            surfel_pos_buf.bind_base(0);
            cell_count_all.bind_base(1);
            surfel_slot_buf.bind_base(2);
            level_idx_all.bind_base(3);
            count_cs->use();
            set_level_uniforms(*count_cs, l);
            gl::dispatch_compute((n + kBlockSize - 1) / kBlockSize, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            // Pass 2: prefix sum (block scan -> block sums scan -> finish).
            const uint32_t nblocks = (levels[l].volume + kBlockSize - 1) / kBlockSize;
            cell_count_all.bind_base(0);
            cell_start_all.bind_base(1);
            block_sum_buf.bind_base(2);
            scan_block_cs->use();
            set_level_uniforms(*scan_block_cs, l);
            {
                GLint u = scan_block_cs->uniform_location("u_n");
                if (u >= 0) scan_block_cs->uniform1ui(u, levels[l].volume);
            }
            gl::dispatch_compute(nblocks, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            block_sum_buf.bind_base(0);
            scan_blocks_cs->use();
            {
                GLint u = scan_blocks_cs->uniform_location("u_nblocks");
                if (u >= 0) scan_blocks_cs->uniform1ui(u, nblocks);
            }
            gl::dispatch_compute(1, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            cell_start_all.bind_base(0);
            block_sum_buf.bind_base(1);
            scan_finish_cs->use();
            set_level_uniforms(*scan_finish_cs, l);
            {
                GLint u = scan_finish_cs->uniform_location("u_n");
                if (u >= 0) scan_finish_cs->uniform1ui(u, levels[l].volume);
            }
            gl::dispatch_compute(nblocks, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            // Pass 3: compact.
            surfel_pos_buf.bind_base(0);
            surfel_slot_buf.bind_base(1);
            cell_start_all.bind_base(2);
            packed_all.bind_base(3);
            level_idx_all.bind_base(4);
            compact_cs->use();
            set_level_uniforms(*compact_cs, l);
            gl::dispatch_compute((n + kBlockSize - 1) / kBlockSize, 1, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
        }

        bind_rt();   // restore raytrace bindings
    };

    // Occupied-cell AABB instance buffer for the debug overlay (all levels).
    gl::Buffer occ_aabb_buf(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    std::vector<glm::vec4> occ_aabbs;
    auto refresh_occupied_cells = [&]() {
        occ_aabbs.clear();
        std::vector<uint32_t> counts(cnt_total);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();
        void* ptr = cell_count_all.map_range(0, size_t(cnt_total) * sizeof(uint32_t), GL_MAP_READ_BIT);
        if (ptr) {
            std::memcpy(counts.data(), ptr, size_t(cnt_total) * sizeof(uint32_t));
            cell_count_all.unmap();
        }
        for (int l = 0; l < kGridLevels; ++l) {
            const LevelSpec& L = levels[l];
            for (uint32_t ci = 0; ci < L.volume; ++ci) {
                if (counts[L.cnt_off + ci] == 0) continue;
                int x = int(ci % uint32_t(L.res.x));
                int y = int((ci / uint32_t(L.res.x)) % uint32_t(L.res.y));
                int z = int(ci / (uint32_t(L.res.x) * uint32_t(L.res.y)));
                glm::vec3 cell_min = L.mn + glm::vec3(x, y, z) * L.cell;
                occ_aabbs.push_back(glm::vec4(cell_min, L.cell));
            }
        }
        occ_aabb_buf.data(occ_aabbs.data(), occ_aabbs.size() * sizeof(glm::vec4));
        for (int l = 0; l < kGridLevels; ++l) {
            uint32_t occ = 0;
            for (uint32_t ci = 0; ci < levels[l].volume; ++ci)
                occ += counts[levels[l].cnt_off + ci] != 0 ? 1u : 0u;
            gllib::logf(gllib::LogLevel::info,
                        "grid L%d: cell=%.4f res=%dx%dx%d surfels=%u occupied=%u",
                        l, levels[l].cell, levels[l].res.x, levels[l].res.y,
                        levels[l].res.z, levels[l].count, occ);
        }
    };

    // Shared rebuild path: everything that depends on the surfel cloud.
    auto rebuild_pipeline = [&]() {
        search_radius = 0.0f;
        for (const auto& s : surfels) search_radius = std::max(search_radius, s.radius);
        glm::vec3 lmn(1e30f), lmx(-1e30f);
        for (const auto& s : surfels) {
            lmn = glm::min(lmn, s.pos);
            lmx = glm::max(lmx, s.pos);
        }
        mn = lmn;
        mx = lmx;
        scene_size = mx - mn;
        upload_surfel_buffers();
        upload_level_buffers();
        build_grid();
        refresh_occupied_cells();
    };

    bind_rt();
    build_grid();
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
    float radius_scale = 1.5f;    // uniform look; near-miss misses retry at 1.9x
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

    // Debug export: SRT_EXPORT=path dumps surfels, cuts, grids and triangles for
    // exact CPU-side replication of the trace (temporary debugging aid).
    if (const char* exp = getenv("SRT_EXPORT")) {
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();
        FILE* f = fopen(exp, "wb");
        if (f) {
            auto w4 = [&](const void* p, size_t bytes) { fwrite(p, 1, bytes, f); };
            const uint32_t magic = 0x32405845;  // "EXP2"
            w4(&magic, 4);
            const uint32_t n32 = uint32_t(N);
            w4(&n32, 4);
            w4(&radius_scale, 4); w4(&search_radius, 4); w4(&leaf_radius, 4);
            w4(&far_plane, 4);
            w4(&light_pos, 12); w4(&light_color, 12);
            w4(&light_intensity, 4); w4(&ambient, 4);
            w4(vp.data(), size_t(N) * 16);
            w4(vn.data(), size_t(N) * 16);
            w4(va.data(), size_t(N) * 16);
            w4(cuts.data(), size_t(N) * kSurfelCuts * 2 * 16);   // anchor form
            const uint32_t nl = uint32_t(kGridLevels);
            w4(&nl, 4);
            for (int li = 0; li < kGridLevels; ++li) {
                w4(&levels[li].mn, 12);
                w4(&levels[li].cell, 4);
                w4(&levels[li].res, 12);
                w4(&levels[li].volume, 4);
                w4(&levels[li].count, 4);
                w4(&levels[li].cnt_off, 4);
                w4(&levels[li].start_off, 4);
                w4(&levels[li].packed_off, 4);
                w4(&levels[li].idx_off, 4);
            }
            w4(&cnt_total, 4); w4(&start_total, 4); w4(&packed_total, 4); w4(&idx_total, 4);
            std::vector<uint32_t> cs(start_total), cc(cnt_total), pi(packed_total), ix(idx_total);
            auto cpy = [&](gl::Buffer& b, std::vector<uint32_t>& dst, size_t elems) {
                if (elems == 0) return;
                void* ptr = b.map_range(0, elems * sizeof(uint32_t), GL_MAP_READ_BIT);
                if (ptr) { std::memcpy(dst.data(), ptr, elems * sizeof(uint32_t)); b.unmap(); }
            };
            cpy(cell_start_all, cs, size_t(start_total));
            cpy(cell_count_all, cc, size_t(cnt_total));
            cpy(packed_all, pi, size_t(packed_total));
            cpy(level_idx_all, ix, size_t(idx_total));
            w4(cs.data(), cs.size() * 4);
            w4(cc.data(), cc.size() * 4);
            w4(pi.data(), pi.size() * 4);
            w4(ix.data(), ix.size() * 4);
            const uint32_t nt = uint32_t(em.tris.size());
            w4(&nt, 4);
            for (const auto& t : em.tris) {
                glm::vec3 fn = glm::normalize(glm::cross(t.p[1] - t.p[0], t.p[2] - t.p[0]));
                w4(&t.p[0], 12); w4(&t.p[1], 12); w4(&t.p[2], 12);
                w4(&fn, 12); w4(&t.n[0], 12); w4(&t.n[1], 12); w4(&t.n[2], 12);
                w4(&t.alb, 12);
            }
            std::vector<uint32_t> mir(em.tris.size());
            for (size_t i = 0; i < em.tris.size(); ++i) mir[i] = em.tri_mirror[i] ? 1u : 0u;
            w4(mir.data(), mir.size() * 4);
            fclose(f);
            gllib::logf(gllib::LogLevel::info, "exported scene to %s", exp);
        }
    }


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
            l = loc("u_light_pos");       if (l >= 0) gbuf->uniform3f(l, light_pos.x, light_pos.y, light_pos.z);
            l = loc("u_light_color");     if (l >= 0) gbuf->uniform3f(l, light_color.x, light_color.y, light_color.z);
            l = loc("u_light_intensity"); if (l >= 0) gbuf->uniform1f(l, light_intensity);
            l = loc("u_ambient");         if (l >= 0) gbuf->uniform1f(l, ambient);

            for (size_t i = 0; i < model.mesh_count(); ++i) {
                // Apply per-mesh transform combined with global model transform
                glm::mat4 combined_xform = model_mat * model.mesh_transform(int(i));
                glm::mat3 normal_mat = glm::transpose(glm::inverse(glm::mat3(combined_xform)));
                l = loc("u_model");           if (l >= 0) gbuf->uniform_matrix4fv(l, glm::value_ptr(combined_xform));
                l = loc("u_normal_mat");      if (l >= 0) gbuf->uniform_matrix3fv(l, glm::value_ptr(normal_mat));
                
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
        {   // per-level grid uniforms
            GLfloat lmin[12], linv[4];
            GLint lres[12];
            GLuint loff[4], lnum[4];
            for (int li = 0; li < kGridLevels; ++li) {
                lmin[li * 3 + 0] = levels[li].mn.x;
                lmin[li * 3 + 1] = levels[li].mn.y;
                lmin[li * 3 + 2] = levels[li].mn.z;
                linv[li] = 1.0f / levels[li].cell;
                lres[li * 3 + 0] = levels[li].res.x;
                lres[li * 3 + 1] = levels[li].res.y;
                lres[li * 3 + 2] = levels[li].res.z;
                loff[li] = levels[li].cnt_off;   // cnt/start/packed offsets share the
                                                 // level index; packed/start have
                                                 // separate uniform arrays below
                lnum[li] = levels[li].count;
            }
            l = loc("u_l_min");      if (l >= 0) glUniform3fv(l, kGridLevels, lmin);
            l = loc("u_l_inv_cell"); if (l >= 0) glUniform1fv(l, kGridLevels, linv);
            l = loc("u_l_res");      if (l >= 0) glUniform3iv(l, kGridLevels, lres);
            l = loc("u_l_num");      if (l >= 0) glUniform1uiv(l, kGridLevels, lnum);
            GLuint soff[4], poff[4];
            for (int li = 0; li < kGridLevels; ++li) {
                soff[li] = levels[li].start_off;
                poff[li] = levels[li].packed_off;
            }
            l = loc("u_l_cnt_off");   if (l >= 0) glUniform1uiv(l, kGridLevels, loff);
            l = loc("u_l_start_off"); if (l >= 0) glUniform1uiv(l, kGridLevels, soff);
            l = loc("u_l_packed_off");if (l >= 0) glUniform1uiv(l, kGridLevels, poff);
            l = loc("u_levels");      if (l >= 0) raytrace->uniform1i(l, kGridLevels);
        }
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

            // Model selection
            static int last_model_index = current_model_index;
            if (ImGui::Combo("Model", &current_model_index, model_names, num_models)) {
                if (current_model_index != last_model_index) {
                    last_model_index = current_model_index;
                    model_path = model_paths[current_model_index];
                    model_mat = get_model_transform(current_model_index);  // Update model transform for the new model

                    // Load new model and extract triangles
                    if (load_model_and_extract(model_path)) {
                        // Rebuild the whole surfel pipeline with the new model
                        build_surfels();
                        rebuild_pipeline();
                        traced = false;
                    }
                }
            }
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
            static float last_base_step = base_step;
            ImGui::SliderFloat("Surfel step (m)", &base_step, 0.005f, 0.2f, "%.4f",
                               ImGuiSliderFlags_Logarithmic);
            if (base_step != last_base_step) {
                last_base_step = base_step;
                build_surfels();
                rebuild_pipeline();
                traced = false;
            }
            ImGui::Text("Leaf radius r = %.5f (coarse)", leaf_radius);
            for (int li = 0; li < kGridLevels; ++li) {
                ImGui::Text("L%d cell %.4f res %dx%dx%d surfels %u",
                            li, levels[li].cell, levels[li].res.x, levels[li].res.y,
                            levels[li].res.z, levels[li].count);
            }
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
#include "surfel_bake.hpp"

#include <gllib/log.hpp>
#include <glad/glad.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <thread>
#include <unordered_map>

namespace mosaic {
namespace {

// Covering radius of a Poisson-disk set: a disk of 1.35 * spacing around every
// sample leaves no hole. Same constant as example 38 (kLATTICE_R).
constexpr float kCoverRadius = 1.35f;
// Below this |dot| between incident face normals an edge is a crease, not
// curvature: the two faces are flat and edge fidelity is not a density problem.
constexpr float kSharpDot = 0.6f;
constexpr float kCurvKappaMax = 200.0f;
// Faces below this curvature use the geometric face normal rather than the
// interpolated vertex normal (robust against junk vertex normals in assets).
constexpr float kFlatKappa = 0.5f;

float srgb_to_linear(float c) {
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

uint32_t quantize_unit(float v, uint32_t bits) {
    const float maxv = float((1u << bits) - 1u);
    return uint32_t(std::lround(std::clamp(v, 0.0f, 1.0f) * maxv));
}

} // namespace

glm::vec3 surfel_position(const SurfelSet& set, const PackedSurfel& p) {
    const glm::vec3 q(float(p.x >> 16), float(p.y >> 16), float(p.y & 0xFFFFu));
    return set.aabb_min + (set.aabb_max - set.aabb_min) * (q / 65535.0f);
}

glm::vec3 surfel_normal(const PackedSurfel& p) {
    auto dec = [](uint32_t u) {
        int v = int(u & 0xFFFFu);
        if (v > 32767) v -= 65536;
        return float(v) / 32767.0f;
    };
    glm::vec2 e(dec(p.w >> 16), dec(p.w));
    glm::vec3 n(e.x, e.y, 1.0f - std::abs(e.x) - std::abs(e.y));
    if (n.z < 0.0f) {
        n.x = (1.0f - std::abs(e.y)) * (e.x >= 0.0f ? 1.0f : -1.0f);
        n.y = (1.0f - std::abs(e.x)) * (e.y >= 0.0f ? 1.0f : -1.0f);
    }
    const float len = glm::length(n);
    return len > 1e-20f ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
}

float surfel_radius(const SurfelSet& set, const PackedSurfel& p) {
    return float(p.x & 0xFFFFu) / 65535.0f * set.radius_scale;
}

uint32_t pack_rgb9e5(const glm::vec3& c) {
    // GL_RGB9_E5: 9-bit mantissas, shared 5-bit exponent with a bias of 15,
    // representable range [0, 65408]. Mirrors the GLSL decoder in surfel.glsl.
    constexpr float kMax = 65408.0f;
    const float r = std::clamp(c.r, 0.0f, kMax);
    const float g = std::clamp(c.g, 0.0f, kMax);
    const float b = std::clamp(c.b, 0.0f, kMax);
    const float maxc = std::max({r, g, b});
    if (maxc < 1e-9f) return 0;

    int exp_shared = std::max(-16, int(std::floor(std::log2(maxc)))) + 1;
    auto denom = [&] { return std::pow(2.0f, float(exp_shared - 15 - 9)); };
    int max_m = int(std::lround(maxc / denom()));
    if (max_m == 512) { ++exp_shared; }          // rounding pushed it to the next exponent
    exp_shared = std::clamp(exp_shared + 15, 0, 31);

    const float d = denom();
    const uint32_t rm = uint32_t(std::clamp(std::lround(r / d), 0L, 511L));
    const uint32_t gm = uint32_t(std::clamp(std::lround(g / d), 0L, 511L));
    const uint32_t bm = uint32_t(std::clamp(std::lround(b / d), 0L, 511L));
    return rm | (gm << 9) | (bm << 18) | (uint32_t(exp_shared) << 27);
}

namespace {

uint32_t pack_oct_snorm16(const glm::vec3& n_in) {
    glm::vec3 n = n_in;
    const float l1 = std::abs(n.x) + std::abs(n.y) + std::abs(n.z);
    if (l1 < 1e-20f) return 0;
    n /= l1;
    glm::vec2 e(n.x, n.y);
    if (n.z < 0.0f) {
        e = glm::vec2((1.0f - std::abs(n.y)) * (n.x >= 0.0f ? 1.0f : -1.0f),
                      (1.0f - std::abs(n.x)) * (n.y >= 0.0f ? 1.0f : -1.0f));
    }
    const int qx = int(std::lround(std::clamp(e.x, -1.0f, 1.0f) * 32767.0f));
    const int qy = int(std::lround(std::clamp(e.y, -1.0f, 1.0f) * 32767.0f));
    return (uint32_t(qx & 0xFFFF) << 16) | uint32_t(qy & 0xFFFF);
}

// --- Triangle extraction ----------------------------------------------------

struct Tri {
    glm::vec3 p[3];
    glm::vec3 n[3];
    glm::vec2 uv[3];
    float area = 0.0f;
    float curvature = 0.0f;   // max smooth-edge dihedral / edge length
    int object = 0;           // connected component id within the mesh
};

// Extracts one mesh in its OWN vertex space (no node transform): the surfel set
// is an asset-resident property of the mesh, shared by every instance of it.
std::vector<Tri> extract_mesh(const gfx::Mesh& mesh) {
    std::vector<Tri> tris;
    const auto& vs = mesh.vertices();
    const auto& is = mesh.indices();

    auto add = [&](const gfx::Vertex& a, const gfx::Vertex& b, const gfx::Vertex& c) {
        Tri t;
        const gfx::Vertex* v[3] = {&a, &b, &c};
        for (int k = 0; k < 3; ++k) {
            t.p[k] = glm::vec3(v[k]->position[0], v[k]->position[1], v[k]->position[2]);
            t.n[k] = glm::vec3(v[k]->normal[0], v[k]->normal[1], v[k]->normal[2]);
            t.uv[k] = glm::vec2(v[k]->texcoord[0], v[k]->texcoord[1]);
            const float nl = glm::length(t.n[k]);
            t.n[k] = nl > 1e-8f ? t.n[k] / nl : glm::vec3(0, 1, 0);
        }
        t.area = 0.5f * glm::length(glm::cross(t.p[1] - t.p[0], t.p[2] - t.p[0]));
        if (t.area < 1e-12f) return;   // degenerate
        tris.push_back(t);
    };

    if (is.empty()) {
        for (size_t v = 0; v + 2 < vs.size(); v += 3) add(vs[v], vs[v + 1], vs[v + 2]);
    } else {
        for (size_t k = 0; k + 2 < is.size(); k += 3) add(vs[is[k]], vs[is[k + 1]], vs[is[k + 2]]);
    }
    return tris;
}

// Fills in per-triangle curvature and connected-component object ids.
//
// Curvature comes from SMOOTH edge dihedrals (kappa = angle / edge length) using
// geometric face normals only, so bad vertex normals cannot inflate it. Sharp
// creases are excluded: those faces are flat and need cut planes, not density.
// Object labelling matters because two surfaces that merely touch must not thin
// each other's samples — that opens holes at seams.
void annotate(std::vector<Tri>& tris) {
    struct EdgeAccum {
        int count = 0;
        int ti[2] = {0, 0};
        glm::vec3 n[2]{};
        float len = 0.0f;
    };
    std::map<std::pair<int64_t, int64_t>, EdgeAccum> edges;

    auto qkey = [](const glm::vec3& p) -> int64_t {
        return int64_t(std::llround(p.x * 10000.0f)) * 73856093 ^
               int64_t(std::llround(p.y * 10000.0f)) * 19349663 ^
               int64_t(std::llround(p.z * 10000.0f)) * 83492791;
    };

    for (size_t ti = 0; ti < tris.size(); ++ti) {
        const Tri& t = tris[ti];
        const glm::vec3 fn = glm::normalize(glm::cross(t.p[1] - t.p[0], t.p[2] - t.p[0]));
        for (int k = 0; k < 3; ++k) {
            const glm::vec3& p0 = t.p[k];
            const glm::vec3& p1 = t.p[(k + 1) % 3];
            auto ka = qkey(p0), kb = qkey(p1);
            if (ka > kb) std::swap(ka, kb);
            EdgeAccum& e = edges[{ka, kb}];
            if (e.count < 2) {
                e.ti[e.count] = int(ti);
                e.n[e.count] = fn;
                e.len = glm::length(p1 - p0);
            }
            ++e.count;
        }
    }

    // Connected components via union-find over shared edges.
    std::vector<int> parent(tris.size());
    for (size_t i = 0; i < parent.size(); ++i) parent[i] = int(i);
    std::function<int(int)> find = [&](int x) {
        while (parent[size_t(x)] != x) {
            parent[size_t(x)] = parent[size_t(parent[size_t(x)])];
            x = parent[size_t(x)];
        }
        return x;
    };
    for (const auto& [key, e] : edges) {
        if (e.count < 2) continue;
        const int a = find(e.ti[0]), b = find(e.ti[1]);
        if (a != b) parent[size_t(a)] = b;
    }
    std::unordered_map<int, int> comp;
    int next_comp = 0;
    for (size_t i = 0; i < tris.size(); ++i) {
        const int r = find(int(i));
        auto it = comp.find(r);
        if (it == comp.end()) it = comp.emplace(r, next_comp++).first;
        tris[i].object = it->second;
    }

    for (const auto& [key, e] : edges) {
        if (e.count != 2) continue;
        const float dn = std::abs(glm::dot(e.n[0], e.n[1]));
        if (dn < kSharpDot) continue;                    // crease, not curvature
        const float ang = std::acos(std::clamp(dn, 0.0f, 1.0f));
        const float kappa = std::min(ang / std::max(e.len, 1e-6f), kCurvKappaMax);
        tris[size_t(e.ti[0])].curvature = std::max(tris[size_t(e.ti[0])].curvature, kappa);
        tris[size_t(e.ti[1])].curvature = std::max(tris[size_t(e.ti[1])].curvature, kappa);
    }
}

// --- Poisson-disk sampling --------------------------------------------------

struct Sample {
    glm::vec3 pos{0.0f};
    glm::vec3 nrm{0.0f};
    glm::vec2 uv{0.0f};
    float radius = 0.0f;
    int object = 0;
};

// Flat 3D cell grid for min-distance queries. cell = the coarsest local spacing,
// so a 27-cell neighbourhood covers every query.
struct SampleGrid {
    glm::vec3 mn{0.0f};
    float cell = 1.0f;
    glm::ivec3 res{1};
    std::vector<std::vector<uint32_t>> cells;

    void setup(const glm::vec3& bmin, const glm::vec3& bmax, float c) {
        cell = std::max(c, 1e-6f);
        mn = bmin - glm::vec3(cell);
        const glm::vec3 mx = bmax + glm::vec3(cell);
        for (int a = 0; a < 3; ++a)
            res[a] = std::max(1, int(std::ceil((mx[a] - mn[a]) / cell)));
        // Guard against a pathological aspect ratio blowing up memory.
        while (double(res.x) * double(res.y) * double(res.z) > 8e6) {
            cell *= 2.0f;
            for (int a = 0; a < 3; ++a)
                res[a] = std::max(1, int(std::ceil((mx[a] - mn[a]) / cell)));
        }
        cells.assign(size_t(res.x) * size_t(res.y) * size_t(res.z), {});
    }
    glm::ivec3 icell(const glm::vec3& p) const {
        return glm::clamp(glm::ivec3(glm::floor((p - mn) / cell)), glm::ivec3(0), res - 1);
    }
    size_t flat(const glm::ivec3& c) const {
        return size_t(c.x) + size_t(res.x) * (size_t(c.y) + size_t(res.y) * size_t(c.z));
    }
    void insert(const glm::vec3& p, uint32_t idx) { cells[flat(icell(p))].push_back(idx); }

    // Rejects only accepted samples on the SAME surface of the SAME component:
    // a box standing on a floor must not thin the floor's samples.
    bool too_close(const glm::vec3& p, const glm::vec3& n, int object, float h,
                   const std::vector<Sample>& pts) const {
        const glm::ivec3 c = icell(p);
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const glm::ivec3 cc = c + glm::ivec3(dx, dy, dz);
            if (glm::any(glm::lessThan(cc, glm::ivec3(0))) ||
                glm::any(glm::greaterThanEqual(cc, res))) continue;
            for (uint32_t i : cells[flat(cc)]) {
                const Sample& s = pts[i];
                if (s.object != object) continue;
                if (glm::dot(n, s.nrm) < 0.85f) continue;
                const float d = std::max(h, s.radius / kCoverRadius);
                const glm::vec3 delta = s.pos - p;
                if (glm::dot(delta, delta) < d * d) return true;
            }
        }
        return false;
    }

    // Nearest same-surface sample, for building LOD parent links.
    uint32_t nearest(const glm::vec3& p, const glm::vec3& n,
                     const std::vector<Sample>& pts, int rings) const {
        const glm::ivec3 c = icell(p);
        uint32_t best = 0xFFFFFFFFu;
        float best_d2 = 1e30f;
        for (int dz = -rings; dz <= rings; ++dz)
        for (int dy = -rings; dy <= rings; ++dy)
        for (int dx = -rings; dx <= rings; ++dx) {
            const glm::ivec3 cc = c + glm::ivec3(dx, dy, dz);
            if (glm::any(glm::lessThan(cc, glm::ivec3(0))) ||
                glm::any(glm::greaterThanEqual(cc, res))) continue;
            for (uint32_t i : cells[flat(cc)]) {
                const Sample& s = pts[i];
                if (glm::dot(n, s.nrm) < 0.0f) continue;
                const glm::vec3 delta = s.pos - p;
                const float d2 = glm::dot(delta, delta);
                if (d2 < best_d2) { best_d2 = d2; best = i; }
            }
        }
        return best;
    }
};

std::vector<Sample> sample_surface(const std::vector<Tri>& tris, float h_base,
                                   const BakeParams& params, uint32_t seed_salt) {
    const size_t ntri = tris.size();
    if (ntri == 0) return {};

    std::vector<float> h_i(ntri), cdf;
    cdf.reserve(ntri);
    glm::vec3 bmin(1e30f), bmax(-1e30f);
    double target = 0.0;
    float total_w = 0.0f;
    for (size_t i = 0; i < ntri; ++i) {
        const Tri& t = tris[i];
        for (int k = 0; k < 3; ++k) { bmin = glm::min(bmin, t.p[k]); bmax = glm::max(bmax, t.p[k]); }
        const float kappa = std::min(t.curvature, kCurvKappaMax);
        const float h = std::max(h_base / (1.0f + params.curvature_adapt * kappa * h_base),
                                 h_base * params.curvature_floor);
        h_i[i] = h;
        const float w = t.area / (h * h);   // this triangle's share of the samples
        total_w += w;
        cdf.push_back(total_w);
        target += double(w);
    }
    if (total_w <= 0.0f) return {};

    const size_t target_count =
        size_t(std::min(target * 1.1, double(params.max_surfels_per_lod)));
    if (target_count == 0) return {};
    const size_t max_attempts = std::max<size_t>(20000, size_t(std::min(target * 4.0, 8e6)));

    std::mt19937 rng(params.seed ^ seed_salt);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);

    std::vector<Sample> pts;
    pts.reserve(target_count);
    SampleGrid grid;
    grid.setup(bmin, bmax, h_base);

    size_t consecutive_fail = 0;
    for (size_t attempt = 0; attempt < max_attempts; ++attempt) {
        const float u = unit(rng) * total_w;
        const size_t ti = size_t(std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
        if (ti >= ntri) continue;
        const Tri& t = tris[ti];

        float a = unit(rng), b = unit(rng);
        if (a + b > 1.0f) { a = 1.0f - a; b = 1.0f - b; }
        const float w0 = 1.0f - a - b;

        Sample s;
        s.pos = t.p[0] * w0 + t.p[1] * a + t.p[2] * b;
        s.uv  = t.uv[0] * w0 + t.uv[1] * a + t.uv[2] * b;
        if (t.curvature < kFlatKappa) {
            s.nrm = glm::normalize(glm::cross(t.p[1] - t.p[0], t.p[2] - t.p[0]));
        } else {
            const glm::vec3 n = t.n[0] * w0 + t.n[1] * a + t.n[2] * b;
            const float nl = glm::length(n);
            s.nrm = nl > 1e-8f ? n / nl
                               : glm::normalize(glm::cross(t.p[1] - t.p[0], t.p[2] - t.p[0]));
        }
        s.object = t.object;
        s.radius = kCoverRadius * h_i[ti];

        if (grid.too_close(s.pos, s.nrm, s.object, h_i[ti], pts)) {
            if (++consecutive_fail >= 4000) break;   // surface is packed
            continue;
        }
        consecutive_fail = 0;
        grid.insert(s.pos, uint32_t(pts.size()));
        pts.push_back(s);
        if (pts.size() >= target_count) break;
    }
    return pts;
}

} // namespace

// --- CpuTexture -------------------------------------------------------------

glm::vec4 CpuTexture::sample(glm::vec2 uv) const {
    if (!valid()) return glm::vec4(1.0f);
    // Repeat wrap, matching the sampler state gfx::Model sets on its textures.
    uv -= glm::floor(uv);
    const float fx = uv.x * float(width) - 0.5f;
    const float fy = uv.y * float(height) - 0.5f;
    const int x0 = int(std::floor(fx)), y0 = int(std::floor(fy));
    const float tx = fx - float(x0), ty = fy - float(y0);
    auto wrap = [](int v, int n) { const int m = v % n; return m < 0 ? m + n : m; };
    auto at = [&](int x, int y) -> const glm::vec4& {
        return texels[size_t(wrap(y, height)) * size_t(width) + size_t(wrap(x, width))];
    };
    const glm::vec4 a = glm::mix(at(x0, y0),     at(x0 + 1, y0),     tx);
    const glm::vec4 b = glm::mix(at(x0, y0 + 1), at(x0 + 1, y0 + 1), tx);
    return glm::mix(a, b, ty);
}

std::vector<CpuTexture> read_back_textures(const gfx::Model& model) {
    std::vector<CpuTexture> out(model.texture_count());
    std::vector<uint8_t> staging;

    for (size_t i = 0; i < model.texture_count(); ++i) {
        const auto& tex = model.texture(i);
        if (!tex || tex->handle() == 0) continue;

        const int w = tex->width(), h = tex->height();
        if (w <= 0 || h <= 0) continue;

        // gfx::Model discards the decoded bytes after upload, so read level 0
        // back out of GL. sRGB internal formats return the STORED (encoded)
        // bytes here — glGetTextureImage performs no sRGB decode — so the
        // linearization below is required, not optional.
        GLint internal_fmt = 0;
        glGetTextureLevelParameteriv(tex->handle(), 0, GL_TEXTURE_INTERNAL_FORMAT, &internal_fmt);
        const bool is_srgb = (internal_fmt == GL_SRGB8 || internal_fmt == GL_SRGB8_ALPHA8);

        staging.resize(size_t(w) * size_t(h) * 4);
        glGetTextureImage(tex->handle(), 0, GL_RGBA, GL_UNSIGNED_BYTE,
                          GLsizei(staging.size()), staging.data());

        CpuTexture& cpu = out[i];
        cpu.width = w;
        cpu.height = h;
        cpu.texels.resize(size_t(w) * size_t(h));
        for (size_t p = 0; p < cpu.texels.size(); ++p) {
            const float r = float(staging[p * 4 + 0]) / 255.0f;
            const float g = float(staging[p * 4 + 1]) / 255.0f;
            const float b = float(staging[p * 4 + 2]) / 255.0f;
            const float a = float(staging[p * 4 + 3]) / 255.0f;
            cpu.texels[p] = is_srgb
                ? glm::vec4(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a)
                : glm::vec4(r, g, b, a);
        }
    }
    return out;
}

// --- bake_mesh --------------------------------------------------------------

SurfelSet bake_mesh(const gfx::Model& model, size_t mesh_index,
                    const std::vector<CpuTexture>& textures,
                    const BakeParams& params, float object_scale) {
    SurfelSet set;

    std::vector<Tri> tris = extract_mesh(model.mesh(mesh_index));
    if (tris.empty()) return set;
    annotate(tris);

    // Material: one per mesh in gfx::Model, so this is resolved once.
    glm::vec3 base_factor(1.0f), emissive_factor(0.0f);
    const CpuTexture* base_tex = nullptr;
    const CpuTexture* emissive_tex = nullptr;
    bool two_sided = false;
    const int mat_idx = model.mesh_material(mesh_index);
    if (mat_idx >= 0 && size_t(mat_idx) < model.material_count()) {
        const gfx::ModelMaterialInfo& m = model.material_info(size_t(mat_idx));
        base_factor = glm::vec3(m.base_color_factor[0], m.base_color_factor[1],
                                m.base_color_factor[2]);
        emissive_factor = glm::vec3(m.emissive_factor[0], m.emissive_factor[1],
                                    m.emissive_factor[2]);
        two_sided = m.double_sided;
        auto pick = [&](int idx) -> const CpuTexture* {
            if (idx < 0 || size_t(idx) >= textures.size()) return nullptr;
            return textures[size_t(idx)].valid() ? &textures[size_t(idx)] : nullptr;
        };
        base_tex = pick(m.base_color_tex);
        emissive_tex = pick(m.emissive_tex);
    }

    // Object-space AABB is the position quantization domain.
    glm::vec3 mn(1e30f), mx(-1e30f);
    for (const Tri& t : tris)
        for (int k = 0; k < 3; ++k) { mn = glm::min(mn, t.p[k]); mx = glm::max(mx, t.p[k]); }
    // A perfectly flat mesh (a Cornell wall) has zero extent on one axis, which
    // would make the quantization scale divide by zero.
    const glm::vec3 extent = glm::max(mx - mn, glm::vec3(1e-5f));
    set.aabb_min = mn;
    set.aabb_max = mn + extent;

    // The set lives in object space, so the world-space spacing target has to be
    // divided by the instance scale that will place it.
    const float base_spacing = params.world_spacing / std::max(object_scale, 1e-6f);

    std::vector<std::vector<Sample>> levels(Config::kSurfelLods);
    std::vector<SampleGrid> grids(Config::kSurfelLods);
    for (int l = 0; l < Config::kSurfelLods; ++l) {
        const float spacing = base_spacing * Config::kLodSpacing[l];
        levels[size_t(l)] = sample_surface(tris, spacing, params, uint32_t(l) * 7919u);
        set.lods[l].spacing = spacing;
    }

    // Radius quantization domain: the largest radius across all levels.
    float max_radius = 1e-6f;
    for (const auto& lv : levels)
        for (const Sample& s : lv) max_radius = std::max(max_radius, s.radius);
    set.radius_scale = max_radius;

    // Neighbour grids per level, used for the parent links below.
    for (int l = 0; l < Config::kSurfelLods; ++l) {
        if (levels[size_t(l)].empty()) continue;
        grids[size_t(l)].setup(mn, set.aabb_max, base_spacing * Config::kLodSpacing[l]);
        for (uint32_t i = 0; i < levels[size_t(l)].size(); ++i)
            grids[size_t(l)].insert(levels[size_t(l)][i].pos, i);
    }

    uint32_t offset = 0;
    std::vector<uint32_t> level_base(Config::kSurfelLods, 0);
    for (int l = 0; l < Config::kSurfelLods; ++l) {
        level_base[size_t(l)] = offset;
        set.lods[l].offset = offset;
        set.lods[l].count = uint32_t(levels[size_t(l)].size());
        offset += set.lods[l].count;
    }
    set.packed.reserve(offset);
    set.parent.reserve(offset);
    set.emissive_dense.reserve(offset);

    for (int l = 0; l < Config::kSurfelLods; ++l) {
        const auto& lv = levels[size_t(l)];
        double radius_sum = 0.0;
        for (const Sample& s : lv) {
            PackedSurfel ps{};
            const glm::vec3 rel = (s.pos - set.aabb_min) / extent;
            const uint32_t px = quantize_unit(rel.x, 16);
            const uint32_t py = quantize_unit(rel.y, 16);
            const uint32_t pz = quantize_unit(rel.z, 16);
            const uint32_t qr = quantize_unit(s.radius / set.radius_scale, 16);

            glm::vec3 albedo = base_factor;
            if (base_tex) albedo *= glm::vec3(base_tex->sample(s.uv));
            glm::vec3 emissive = emissive_factor;
            if (emissive_tex) emissive *= glm::vec3(emissive_tex->sample(s.uv));

            uint32_t flags = 0;
            if (two_sided) flags |= kSurfelTwoSided;
            const float emissive_lum = glm::dot(emissive, glm::vec3(0.2126f, 0.7152f, 0.0722f));
            if (emissive_lum > 1e-4f) {
                flags |= kSurfelEmissive;
                set.emissive.push_back({uint32_t(set.packed.size()),
                                        emissive.r, emissive.g, emissive.b});
            } else {
                emissive = glm::vec3(0.0f);
            }
            set.emissive_dense.push_back(pack_rgb9e5(emissive));

            ps.x = (px << 16) | qr;
            ps.y = (py << 16) | pz;
            ps.z = (quantize_unit(albedo.r, 8) << 24) |
                   (quantize_unit(albedo.g, 8) << 16) |
                   (quantize_unit(albedo.b, 8) << 8) | (flags & 0xFFu);
            ps.w = pack_oct_snorm16(s.nrm);
            set.packed.push_back(ps);

            // Parent = nearest same-facing sample one level coarser. Searching
            // two rings covers the coarser level's own spacing.
            uint32_t parent = 0xFFFFFFFFu;
            if (l + 1 < Config::kSurfelLods && !levels[size_t(l + 1)].empty()) {
                const uint32_t p = grids[size_t(l + 1)].nearest(s.pos, s.nrm,
                                                                levels[size_t(l + 1)], 2);
                if (p != 0xFFFFFFFFu) parent = level_base[size_t(l + 1)] + p;
            }
            set.parent.push_back(parent);
            radius_sum += double(s.radius);
        }
        set.lods[l].mean_radius = lv.empty() ? 0.0f : float(radius_sum / double(lv.size()));
    }

    return set;
}

double scene_surface_area(const std::vector<Instance>& instances,
                          const std::vector<const gfx::Model*>& models) {
    // Per-mesh object-space area, computed once and reused across instances.
    std::map<std::pair<int, int>, double> mesh_area;
    double total = 0.0;

    for (const Instance& inst : instances) {
        if (inst.model < 0 || size_t(inst.model) >= models.size()) continue;
        const gfx::Model& model = *models[size_t(inst.model)];
        if (size_t(inst.mesh) >= model.mesh_count()) continue;

        const auto key = std::make_pair(inst.model, inst.mesh);
        auto it = mesh_area.find(key);
        if (it == mesh_area.end()) {
            double area = 0.0;
            const gfx::Mesh& mesh = model.mesh(size_t(inst.mesh));
            const auto& vs = mesh.vertices();
            const auto& is = mesh.indices();
            auto tri_area = [&](const gfx::Vertex& a, const gfx::Vertex& b, const gfx::Vertex& c) {
                const glm::vec3 pa(a.position[0], a.position[1], a.position[2]);
                const glm::vec3 pb(b.position[0], b.position[1], b.position[2]);
                const glm::vec3 pc(c.position[0], c.position[1], c.position[2]);
                return 0.5 * double(glm::length(glm::cross(pb - pa, pc - pa)));
            };
            if (is.empty()) {
                for (size_t v = 0; v + 2 < vs.size(); v += 3)
                    area += tri_area(vs[v], vs[v + 1], vs[v + 2]);
            } else {
                for (size_t k = 0; k + 2 < is.size(); k += 3)
                    area += tri_area(vs[is[k]], vs[is[k + 1]], vs[is[k + 2]]);
            }
            it = mesh_area.emplace(key, area).first;
        }
        // Area scales with the square of a uniform transform scale.
        const float s = glm::length(glm::vec3(inst.xform[0]));
        total += it->second * double(s) * double(s);
    }
    return total;
}

// --- SurfelLibrary ----------------------------------------------------------

void SurfelLibrary::build(const std::vector<Instance>& instances,
                          const std::vector<const gfx::Model*>& models,
                          const std::vector<std::string>& model_paths,
                          const BakeParams& params) {
    const auto t_start = std::chrono::steady_clock::now();
    entries_.clear();
    lookup_.clear();
    total_surfels_ = 0;
    total_emissive_ = 0;
    cached_sets_ = 0;

    // 1. Collect the distinct (model, mesh) pairs the instances reference. The
    //    object scale comes from the first instance that places the mesh; a set
    //    is shared by every instance of it, which is the point of object space.
    for (const Instance& inst : instances) {
        if (inst.model < 0 || size_t(inst.model) >= models.size()) continue;
        const gfx::Model& model = *models[size_t(inst.model)];
        if (size_t(inst.mesh) >= model.mesh_count()) continue;
        const auto key = std::make_pair(inst.model, inst.mesh);
        if (lookup_.count(key)) continue;

        Entry e;
        e.model = inst.model;
        e.mesh = inst.mesh;
        const glm::vec3 col_scale(glm::length(glm::vec3(inst.xform[0])),
                                  glm::length(glm::vec3(inst.xform[1])),
                                  glm::length(glm::vec3(inst.xform[2])));
        // Anisotropy check: the object->world scale is assumed uniform, both
        // here and in the debug point-size computation. Report violations
        // instead of letting them show up as surfels of inconsistent world size.
        const float aniso = std::max({col_scale.x, col_scale.y, col_scale.z}) /
                            std::max(std::min({col_scale.x, col_scale.y, col_scale.z}), 1e-20f);
        if (aniso > 1.05f)
            gllib::logf(gllib::LogLevel::warn,
                        "instance (model %d, mesh %d) has non-uniform scale "
                        "(%.5f, %.5f, %.5f), ratio %.2f: surfel sizing assumes uniform",
                        inst.model, inst.mesh, col_scale.x, col_scale.y, col_scale.z, aniso);
        // Use the geometric mean so a moderate anisotropy degrades gracefully
        // rather than biasing every surfel toward the X axis.
        e.object_scale = std::cbrt(std::max(col_scale.x * col_scale.y * col_scale.z, 1e-30f));
        lookup_[key] = int(entries_.size());
        entries_.push_back(std::move(e));
    }

    auto path_of = [&](int model_index) -> std::string {
        return size_t(model_index) < model_paths.size() ? model_paths[size_t(model_index)]
                                                        : std::string();
    };

    // 2. Cache probe (cheap, serial).
    std::vector<size_t> misses;
    for (size_t i = 0; i < entries_.size(); ++i) {
        Entry& e = entries_[i];
        const std::string path = path_of(e.model);
        if (load_cache(cache_path(path, size_t(e.mesh)), e.set, path,
                       *models[size_t(e.model)], size_t(e.mesh), params, e.object_scale)) {
            ++cached_sets_;
        } else {
            misses.push_back(i);
        }
    }

    // 3. Texture read-back must happen on this thread: it issues GL calls, and
    //    the GL context is not current on the worker threads.
    std::map<int, std::vector<CpuTexture>> textures;
    for (size_t i : misses) {
        const int m = entries_[i].model;
        if (!textures.count(m)) textures.emplace(m, read_back_textures(*models[size_t(m)]));
    }

    // 4. Bake the misses in parallel. Each mesh is independent, and Sponza has
    //    103 of them, so this is the difference between a usable first run and a
    //    ten-minute stall.
    if (!misses.empty()) {
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const unsigned nthreads = std::min<unsigned>(hw, unsigned(misses.size()));
        std::atomic<size_t> next{0};
        std::atomic<size_t> done{0};

        auto worker = [&]() {
            for (;;) {
                const size_t k = next.fetch_add(1);
                if (k >= misses.size()) return;
                Entry& e = entries_[misses[k]];
                e.set = bake_mesh(*models[size_t(e.model)], size_t(e.mesh),
                                  textures[e.model], params, e.object_scale);
                done.fetch_add(1);
            }
        };

        gllib::logf(gllib::LogLevel::info, "surfel bake: %zu set(s) to build on %u threads",
                    misses.size(), nthreads);
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (unsigned t = 0; t < nthreads; ++t) pool.emplace_back(worker);
        for (auto& t : pool) t.join();

        // 5. Cache writes, serial (one open file at a time keeps the failure
        //    mode simple, and this is not the expensive part).
        for (size_t i : misses) {
            Entry& e = entries_[i];
            if (e.set.total() == 0) continue;
            const std::string path = path_of(e.model);
            const std::string cpath = cache_path(path, size_t(e.mesh));
            if (!save_cache(cpath, e.set, path, *models[size_t(e.model)],
                            size_t(e.mesh), params, e.object_scale))
                gllib::logf(gllib::LogLevel::warn, "could not write surfel cache '%s'",
                            cpath.c_str());
        }
    }

    // 6. Drop empty sets, then concatenate. Emissive side-array indices are
    //    rebased to the shared buffer so a consumer never needs to know which
    //    set a surfel came from.
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                  [](const Entry& e) { return e.set.total() == 0; }),
                   entries_.end());
    lookup_.clear();
    for (size_t i = 0; i < entries_.size(); ++i)
        lookup_[{entries_[i].model, entries_[i].mesh}] = int(i);

    std::vector<PackedSurfel> packed;
    std::vector<uint32_t> parent;
    std::vector<uint32_t> emissive_dense;
    std::vector<EmissiveSurfel> emissive;
    for (Entry& e : entries_) {
        e.base = uint32_t(packed.size());
        packed.insert(packed.end(), e.set.packed.begin(), e.set.packed.end());
        emissive_dense.insert(emissive_dense.end(),
                              e.set.emissive_dense.begin(), e.set.emissive_dense.end());
        for (uint32_t p : e.set.parent)
            parent.push_back(p == 0xFFFFFFFFu ? p : p + e.base);
        for (EmissiveSurfel es : e.set.emissive) {
            es.index += e.base;
            emissive.push_back(es);
        }
    }
    total_surfels_ = uint32_t(packed.size());
    total_emissive_ = uint32_t(emissive.size());

    if (!packed.empty()) {
        packed_.data(packed.data(), packed.size() * sizeof(PackedSurfel));
        parent_.data(parent.data(), parent.size() * sizeof(uint32_t));
        emissive_dense_.data(emissive_dense.data(), emissive_dense.size() * sizeof(uint32_t));
    }
    if (!emissive.empty())
        emissive_.data(emissive.data(), emissive.size() * sizeof(EmissiveSurfel));

    bake_seconds_ = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();

    gllib::logf(gllib::LogLevel::info,
                "surfel bake: %zu sets (%u from cache), %u surfels, %u emissive, %.2f s",
                entries_.size(), cached_sets_, total_surfels_, total_emissive_, bake_seconds_);

    {
        float smin = 1e30f, smax = 0.0f;
        for (const Entry& e : entries_) {
            smin = std::min(smin, e.object_scale);
            smax = std::max(smax, e.object_scale);
        }
        gllib::logf(gllib::LogLevel::debug, "  object scale range: %.6f .. %.6f",
                    smin, smax);
    }
}

int SurfelLibrary::entry_for(const Instance& inst) const {
    const auto it = lookup_.find({inst.model, inst.mesh});
    return it == lookup_.end() ? -1 : it->second;
}

// --- Disk cache -------------------------------------------------------------

namespace {

constexpr uint32_t kCacheMagic = 0x4D534146u;   // "MSAF"
constexpr uint32_t kCacheVersion = 3u;

uint64_t file_stamp(const std::string& path) {
    std::error_code ec;
    const auto sz = std::filesystem::file_size(path, ec);
    if (ec) return 0;
    const auto mt = std::filesystem::last_write_time(path, ec);
    if (ec) return 0;
    return uint64_t(sz) ^ uint64_t(mt.time_since_epoch().count());
}

uint64_t material_hash(const gfx::Model& model, size_t mesh_index) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void* data, size_t n) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    };
    const int mi = model.mesh_material(mesh_index);
    mix(&mi, sizeof(mi));
    if (mi >= 0 && size_t(mi) < model.material_count()) {
        const gfx::ModelMaterialInfo& m = model.material_info(size_t(mi));
        mix(m.base_color_factor, sizeof(m.base_color_factor));
        mix(m.emissive_factor, sizeof(m.emissive_factor));
        mix(&m.base_color_tex, sizeof(m.base_color_tex));
        mix(&m.emissive_tex, sizeof(m.emissive_tex));
        mix(&m.double_sided, sizeof(m.double_sided));
    }
    return h;
}

struct CacheHeader {
    uint32_t magic, version;
    uint64_t model_stamp;
    uint64_t material_hash;
    uint32_t mesh_index;
    uint32_t lod_count;
    float world_spacing, curvature_adapt, curvature_floor, object_scale;
    uint32_t seed;
    float aabb_min[3], aabb_max[3], radius_scale;
    uint32_t total, emissive_count;
};

template <typename T>
void write_vec(std::ofstream& f, const std::vector<T>& v) {
    const uint64_t n = v.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    if (n) f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(n * sizeof(T)));
}

template <typename T>
bool read_vec(std::ifstream& f, std::vector<T>& v) {
    uint64_t n = 0;
    if (!f.read(reinterpret_cast<char*>(&n), sizeof(n))) return false;
    if (n > (1ull << 32)) return false;
    v.resize(size_t(n));
    if (n && !f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T))))
        return false;
    return true;
}

} // namespace

std::string cache_path(const std::string& model_path, size_t mesh_index) {
    std::filesystem::path p(model_path);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "_m%zu.bin", mesh_index);
    return "cache/39_" + p.stem().string() + buf;
}

bool save_cache(const std::string& path, const SurfelSet& set,
                const std::string& model_path, const gfx::Model& model,
                size_t mesh_index, const BakeParams& params, float object_scale) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    CacheHeader h{};
    h.magic = kCacheMagic;
    h.version = kCacheVersion;
    h.model_stamp = file_stamp(model_path);
    h.material_hash = material_hash(model, mesh_index);
    h.mesh_index = uint32_t(mesh_index);
    h.lod_count = Config::kSurfelLods;
    h.world_spacing = params.world_spacing;
    h.object_scale = object_scale;
    h.curvature_adapt = params.curvature_adapt;
    h.curvature_floor = params.curvature_floor;
    h.seed = params.seed;
    std::memcpy(h.aabb_min, &set.aabb_min, sizeof(h.aabb_min));
    std::memcpy(h.aabb_max, &set.aabb_max, sizeof(h.aabb_max));
    h.radius_scale = set.radius_scale;
    h.total = set.total();
    h.emissive_count = uint32_t(set.emissive.size());
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    f.write(reinterpret_cast<const char*>(set.lods), sizeof(set.lods));
    write_vec(f, set.packed);
    write_vec(f, set.parent);
    write_vec(f, set.emissive_dense);
    write_vec(f, set.emissive);
    return bool(f);
}

bool load_cache(const std::string& path, SurfelSet& out,
                const std::string& model_path, const gfx::Model& model,
                size_t mesh_index, const BakeParams& params, float object_scale) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    CacheHeader h{};
    if (!f.read(reinterpret_cast<char*>(&h), sizeof(h))) return false;
    auto reject = [&](const char* why) {
        gllib::logf(gllib::LogLevel::debug, "surfel cache miss (%s): %s", why, path.c_str());
        return false;
    };
    if (h.magic != kCacheMagic || h.version != kCacheVersion) return reject("version");
    if (h.model_stamp != file_stamp(model_path)) return reject("model stamp");
    if (h.material_hash != material_hash(model, mesh_index)) return reject("material");
    if (h.mesh_index != uint32_t(mesh_index)) return reject("mesh index");
    if (h.lod_count != Config::kSurfelLods) return reject("lod count");
    if (h.world_spacing != params.world_spacing) return reject("spacing");
    if (h.object_scale != object_scale) return reject("object scale");
    if (h.curvature_adapt != params.curvature_adapt ||
        h.curvature_floor != params.curvature_floor ||
        h.seed != params.seed) return reject("bake params");

    std::memcpy(&out.aabb_min, h.aabb_min, sizeof(h.aabb_min));
    std::memcpy(&out.aabb_max, h.aabb_max, sizeof(h.aabb_max));
    out.radius_scale = h.radius_scale;
    if (!f.read(reinterpret_cast<char*>(out.lods), sizeof(out.lods))) return false;
    if (!read_vec(f, out.packed)) return false;
    if (!read_vec(f, out.parent)) return false;
    if (!read_vec(f, out.emissive_dense)) return false;
    if (!read_vec(f, out.emissive)) return false;
    if (out.packed.size() != h.total || out.parent.size() != h.total ||
        out.emissive_dense.size() != h.total) return false;
    return true;
}

} // namespace mosaic

#include "surfels.hpp"

#include <gllib/log.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace sgi {

// --- Triangle extraction ----------------------------------------------------

std::vector<Tri> extract_triangles(const gfx::Model& model) {
    std::vector<Tri> out;

    for (std::size_t mi = 0; mi < model.mesh_count(); ++mi) {
        const gfx::Mesh& mesh = model.mesh(mi);
        const glm::mat4  xf   = model.mesh_transform(mi);
        const glm::mat3  nxf  = glm::transpose(glm::inverse(glm::mat3(xf)));

        glm::vec3 albedo(1.0f), emission(0.0f);
        bool two_sided = false;
        const int mat = model.mesh_material(mi);
        if (mat >= 0 && std::size_t(mat) < model.material_count()) {
            const gfx::ModelMaterialInfo& m = model.material_info(std::size_t(mat));
            albedo = glm::vec3(m.base_color_factor[0], m.base_color_factor[1],
                               m.base_color_factor[2]);
            // emissive_factor already carries KHR_materials_emissive_strength;
            // CornellBoxOriginal.glb authors its ceiling panel at strength 17,
            // so dropping the extension renders it 17x too dim.
            emission = glm::vec3(m.emissive_factor[0], m.emissive_factor[1],
                                 m.emissive_factor[2]);
            two_sided = m.double_sided;
        }

        const std::vector<gfx::Vertex>& vs = mesh.vertices();
        const std::vector<unsigned int>& is = mesh.indices();

        auto emit = [&](const gfx::Vertex& a, const gfx::Vertex& b, const gfx::Vertex& c) {
            Tri t;
            const glm::vec3 lp[3] = {
                glm::vec3(a.position[0], a.position[1], a.position[2]),
                glm::vec3(b.position[0], b.position[1], b.position[2]),
                glm::vec3(c.position[0], c.position[1], c.position[2]),
            };
            for (int k = 0; k < 3; ++k) t.p[k] = glm::vec3(xf * glm::vec4(lp[k], 1.0f));

            const glm::vec3 cr = glm::cross(t.p[1] - t.p[0], t.p[2] - t.p[0]);
            const float len = glm::length(cr);
            if (len < 1e-12f) return;                 // degenerate
            t.area = 0.5f * len;

            // Geometric normal, oriented to agree with the shading normal so a
            // clockwise-wound triangle does not end up emitting backwards.
            glm::vec3 gn = cr / len;
            const glm::vec3 sn = nxf * glm::vec3(a.normal[0], a.normal[1], a.normal[2]);
            if (glm::dot(gn, sn) < 0.0f) gn = -gn;
            t.n = gn;

            t.albedo = albedo;
            t.emission = emission;
            t.double_sided = two_sided;
            out.push_back(t);
        };

        if (is.empty()) {
            for (std::size_t v = 0; v + 2 < vs.size(); v += 3) emit(vs[v], vs[v + 1], vs[v + 2]);
        } else {
            for (std::size_t k = 0; k + 2 < is.size(); k += 3)
                emit(vs[is[k]], vs[is[k + 1]], vs[is[k + 2]]);
        }
    }
    return out;
}

double total_area(const std::vector<Tri>& tris) {
    double a = 0.0;
    for (const Tri& t : tris) a += double(t.area);
    return a;
}

// --- Packing ----------------------------------------------------------------

static glm::vec2 oct_encode(glm::vec3 n) {
    n /= (std::abs(n.x) + std::abs(n.y) + std::abs(n.z));
    if (n.z < 0.0f) {
        const float x = (1.0f - std::abs(n.y)) * (n.x >= 0.0f ? 1.0f : -1.0f);
        const float y = (1.0f - std::abs(n.x)) * (n.y >= 0.0f ? 1.0f : -1.0f);
        n.x = x; n.y = y;
    }
    return glm::vec2(n.x, n.y);
}

uint32_t pack_oct_snorm16(const glm::vec3& n) {
    const glm::vec2 e = glm::clamp(oct_encode(glm::normalize(n)), glm::vec2(-1.0f), glm::vec2(1.0f));
    const int qx = int(std::lround(e.x * 32767.0f));
    const int qy = int(std::lround(e.y * 32767.0f));
    return (uint32_t(qx & 0xFFFF) << 16) | uint32_t(qy & 0xFFFF);
}

glm::vec3 unpack_oct_snorm16(uint32_t p) {
    auto dec = [](uint32_t h) {
        int v = int(h & 0xFFFFu);
        if (v > 32767) v -= 65536;
        return float(v) / 32767.0f;
    };
    const glm::vec2 e(dec(p >> 16), dec(p));
    glm::vec3 n(e.x, e.y, 1.0f - std::abs(e.x) - std::abs(e.y));
    if (n.z < 0.0f) {
        n.x = (1.0f - std::abs(e.y)) * (e.x >= 0.0f ? 1.0f : -1.0f);
        n.y = (1.0f - std::abs(e.x)) * (e.y >= 0.0f ? 1.0f : -1.0f);
    }
    const float len = glm::length(n);
    return len > 1e-20f ? n / len : glm::vec3(0.0f, 1.0f, 0.0f);
}

uint32_t pack_rgb8(const glm::vec3& c, uint32_t flags) {
    const glm::vec3 q = glm::clamp(c, glm::vec3(0.0f), glm::vec3(1.0f)) * 255.0f;
    return uint32_t(std::lround(q.r))
         | (uint32_t(std::lround(q.g)) << 8)
         | (uint32_t(std::lround(q.b)) << 16)
         | ((flags & 0xFFu) << 24);
}

// GL_RGB9_E5: three 9-bit mantissas and a shared 5-bit exponent biased by 15.
// Decoded as mantissa * 2^(biased - 15 - 9), which shaders/common/pack.glsl
// mirrors exactly.
//
// Note this is NOT example 39's routine transcribed: 39 measures the rounding
// overflow (`mantissa == 512`) against the UNBIASED exponent while its divisor
// lambda has already been rebased, so that branch is dead there. The decode
// still round-trips, so it never showed up; here the check is live.
uint32_t pack_rgb9e5(const glm::vec3& c) {
    constexpr float kMax = 65408.0f;
    const glm::vec3 v = glm::clamp(c, glm::vec3(0.0f), glm::vec3(kMax));
    const float maxc = std::max({v.r, v.g, v.b});
    if (!(maxc > 1e-9f)) return 0u;

    int e = std::max(-16, int(std::floor(std::log2(maxc)))) + 1;
    float d = std::exp2(float(e - 9));
    if (int(std::lround(maxc / d)) == 512) { e += 1; d = std::exp2(float(e - 9)); }

    const uint32_t biased = uint32_t(std::clamp(e + 15, 0, 31));
    auto m = [&](float x) { return uint32_t(std::clamp<long>(std::lround(x / d), 0, 511)); };
    return m(v.r) | (m(v.g) << 9) | (m(v.b) << 18) | (biased << 27);
}

glm::vec3 unpack_rgb9e5(uint32_t v) {
    const float e = std::exp2(float(int(v >> 27) - 15 - 9));
    return glm::vec3(float(v & 0x1FFu), float((v >> 9) & 0x1FFu), float((v >> 18) & 0x1FFu)) * e;
}

// --- Sampling ---------------------------------------------------------------

namespace {

// R2, the two-dimensional low-discrepancy sequence built on the plastic number.
// Cheaper than Halton, no permutation tables, and its discrepancy is better
// than a Sobol pair at these counts.
constexpr double kR2A1 = 0.7548776662466927;   // 1/g
constexpr double kR2A2 = 0.5698402909980532;   // 1/g^2

inline float fract(double x) { return float(x - std::floor(x)); }

inline uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

} // namespace

void SurfelSet::build(const std::vector<Tri>& tris, uint32_t target_count, uint32_t seed) {
    const auto t0 = std::chrono::steady_clock::now();

    pos_rad_.clear(); normal_.clear(); albedo_.clear(); emission_.clear();
    tri_of_.clear();
    bounds_ = Bounds{};
    emissive_count_ = 0;
    two_sided_count_ = 0;

    area_ = total_area(tris);
    if (area_ <= 0.0 || target_count == 0) {
        count_ = 0;
        return;
    }

    // Area-weighted allocation, with a floor of one surfel per triangle so a
    // small face is never unrepresented (which would read as a hole in the
    // lighting rather than as coarseness).
    const std::size_t ntri = tris.size();
    std::vector<uint32_t> per_tri(ntri);
    std::size_t total = 0;
    for (std::size_t i = 0; i < ntri; ++i) {
        const double share = double(target_count) * double(tris[i].area) / area_;
        per_tri[i] = uint32_t(std::max<long long>(1, std::llround(share)));
        total += per_tri[i];
    }

    pos_rad_.reserve(total); normal_.reserve(total);
    albedo_.reserve(total);  emission_.reserve(total);
    tri_of_.reserve(total);

    // One global radius, chosen so the discs tile the surface exactly:
    // N * pi * r^2 == A. Uniform radius is deliberate at this stage -- it makes
    // total surfel area an exact invariant and removes a whole class of "is
    // this a radius bug?" from the reference.
    radius_  = float(std::sqrt(area_ / (3.14159265358979 * double(total))));
    spacing_ = float(std::sqrt(area_ / double(total)));

    for (std::size_t i = 0; i < ntri; ++i) {
        const Tri& t = tris[i];
        const glm::vec3 e1 = t.p[1] - t.p[0];
        const glm::vec3 e2 = t.p[2] - t.p[0];

        const uint32_t n_alb = pack_rgb8(t.albedo,
            (glm::dot(t.emission, t.emission) > 0.0f ? kSurfelEmissive : 0u) |
            (t.double_sided ? kSurfelDoubleSided : 0u));
        if (t.double_sided) two_sided_count_ += per_tri[i];
        const uint32_t n_emi = pack_rgb9e5(t.emission);
        const uint32_t n_nrm = pack_oct_snorm16(t.n);
        const bool emissive = glm::dot(t.emission, t.emission) > 0.0f;

        // Cranley-Patterson rotation per triangle: keeps R2's low discrepancy
        // while decorrelating neighbouring faces, so the two triangles of a
        // Cornell wall do not share a visible sample pattern.
        const uint32_t h = hash_u32(seed ^ uint32_t(i * 2654435761u));
        const double ox = double(h & 0xFFFFu) / 65536.0;
        const double oy = double((h >> 16) & 0xFFFFu) / 65536.0;

        for (uint32_t s = 0; s < per_tri[i]; ++s) {
            float a = fract(ox + kR2A1 * double(s + 1));
            float b = fract(oy + kR2A2 * double(s + 1));
            if (a + b > 1.0f) { a = 1.0f - a; b = 1.0f - b; }   // fold into the triangle

            const glm::vec3 p = t.p[0] + e1 * a + e2 * b;
            pos_rad_.push_back(glm::vec4(p, radius_));
            tri_of_.push_back(uint32_t(i));
            normal_.push_back(n_nrm);
            albedo_.push_back(n_alb);
            emission_.push_back(n_emi);
            bounds_.add(p);
            if (emissive) ++emissive_count_;
        }
    }

    count_ = uint32_t(pos_rad_.size());

    // bf_micro.comp's winner key is depth16 | index16, and the index is LOCAL to
    // the U-list whenever there is one -- so the set size is not what has to fit
    // in sixteen bits, the candidate list is. Truncating here instead is what
    // left Sponza at coverage 0.229 (finding 49).
    //
    // The all-pairs path has no candidate list and does still store a global
    // index. Solver::dispatch refuses that combination rather than aliasing
    // winners onto the wrong surfel; this only warns, because whether it matters
    // depends on a config the bake cannot see.
    if (count_ > 65536u)
        gllib::logf(gllib::LogLevel::info,
                    "surfel count %u exceeds 65536: the microbuffer needs its U-list "
                    "path (SGI_NEAR > 0) above that, the all-pairs path cannot index it",
                    count_);
    upload();

    bake_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    gllib::logf(gllib::LogLevel::info,
                "surfels: %u over %zu triangles, area %.4f, radius %.5f, spacing %.5f, "
                "%u emissive, %u double-sided, %.1f KB, %.3f s",
                count_, ntri, area_, radius_, spacing_, emissive_count_, two_sided_count_,
                double(bytes()) / 1024.0, bake_seconds_);
}

void SurfelSet::build_explicit(const std::vector<glm::vec4>& pos_rad,
                               const std::vector<glm::vec3>& normals,
                               const std::vector<glm::vec3>& albedos,
                               const std::vector<glm::vec3>& emissions) {
    count_ = uint32_t(pos_rad.size());
    pos_rad_ = pos_rad;
    normal_.resize(count_); albedo_.resize(count_); emission_.resize(count_);
    tri_of_.clear();   // a hand-built set has no source mesh, so no cut planes
    bounds_ = Bounds{};
    emissive_count_ = 0;
    area_ = 0.0;
    radius_ = count_ ? pos_rad[0].w : 0.0f;

    for (uint32_t i = 0; i < count_; ++i) {
        const bool emissive = glm::dot(emissions[i], emissions[i]) > 0.0f;
        normal_[i]   = pack_oct_snorm16(normals[i]);
        albedo_[i]   = pack_rgb8(albedos[i], emissive ? kSurfelEmissive : 0u);
        emission_[i] = pack_rgb9e5(emissions[i]);
        bounds_.add(glm::vec3(pos_rad[i]));
        area_ += 3.14159265358979 * double(pos_rad[i].w) * double(pos_rad[i].w);
        if (emissive) ++emissive_count_;
    }
    spacing_ = count_ ? float(std::sqrt(area_ / double(count_))) : 0.0f;
    upload();
}

void SurfelSet::upload() {
    if (count_ == 0) return;
    b_pos_rad_.data(pos_rad_.data(), pos_rad_.size() * sizeof(glm::vec4));
    b_normal_.data(normal_.data(), normal_.size() * sizeof(uint32_t));
    b_albedo_.data(albedo_.data(), albedo_.size() * sizeof(uint32_t));
    b_emission_.data(emission_.data(), emission_.size() * sizeof(uint32_t));

    // Irradiance stays fp32 rather than the RGB9E5 of section 1.1. The point of
    // this stage is that a discrepancy against the reference is a bug, and a
    // 9-bit mantissa would put a floor under that. At 30k surfels the two
    // buffers are under a megabyte. Switch to RGB9E5 when the grid lands, and
    // assert the delta then.
    const std::vector<glm::vec4> zero(count_, glm::vec4(0.0f));
    b_irrad_.data(zero.data(), zero.size() * sizeof(glm::vec4));
    b_irrad_prev_.data(zero.data(), zero.size() * sizeof(glm::vec4));
}

void SurfelSet::reset_irradiance() {
    if (count_ == 0) return;
    const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    b_irrad_.clear(GL_RGBA32F, GL_RGBA, GL_FLOAT, zero);
    b_irrad_prev_.clear(GL_RGBA32F, GL_RGBA, GL_FLOAT, zero);
}

void SurfelSet::bind() const {
    b_pos_rad_.bind_base(kBindPosRad);
    b_normal_.bind_base(kBindNormal);
    b_albedo_.bind_base(kBindAlbedo);
    b_emission_.bind_base(kBindEmission);
    b_irrad_.bind_base(kBindIrrad);
    b_irrad_prev_.bind_base(kBindIrradPrev);
}

void SurfelSet::swap_irradiance() {
    std::swap(b_irrad_, b_irrad_prev_);
}

void SurfelSet::carry_irradiance() {
    if (count_ == 0) return;
    glCopyNamedBufferSubData(b_irrad_prev_.handle(), b_irrad_.handle(), 0, 0,
                             GLsizeiptr(count_) * GLsizeiptr(sizeof(glm::vec4)));
}

std::size_t SurfelSet::bytes() const {
    return std::size_t(count_) * (sizeof(glm::vec4) * 3 + sizeof(uint32_t) * 3);
}

std::vector<glm::vec4> SurfelSet::read_irradiance() const {
    std::vector<glm::vec4> out(count_);
    if (count_ == 0) return out;
    glGetNamedBufferSubData(b_irrad_.handle(), 0,
                            GLsizeiptr(count_) * GLsizeiptr(sizeof(glm::vec4)), out.data());
    return out;
}

} // namespace sgi

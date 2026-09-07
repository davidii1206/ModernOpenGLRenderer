#include "direct.hpp"

#include <gllib/log.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace mosaic {
namespace {

// Fit-quality gates, following the spec's coverage/variance tests.
//
// kMinCoverage is deliberately loose. A rectangle is a poor shape for a disc or
// an L, but a poor SHAPE is not a poor LIGHT: scaling the radiance by the
// coverage keeps the emitted power exact whatever the outline, and the error
// left is only that the light is spatially smeared over the bounding rectangle.
// At the distances an area light matters that is invisible, whereas rejecting
// the island entirely sends it back through the gather and costs the banding
// this whole path exists to remove.
constexpr float kMinCoverage = 0.35f;
constexpr uint32_t kMinMembers = 6;
// Connectivity: within this many spacings AND facing the same way. Two and a
// half rather than one keeps an island whole across the gaps a Poisson-disk set
// leaves, without bridging to a panel on the opposite wall.
constexpr float kLinkSpacings = 2.5f;
constexpr float kLinkNormalDot = 0.9f;

struct HashGrid {
    float cell = 1.0f;
    std::unordered_map<uint64_t, std::vector<uint32_t>> buckets;

    static uint64_t key(int x, int y, int z) {
        return (uint64_t(uint32_t(x)) * 0x9E3779B97F4A7C15ull) ^
               (uint64_t(uint32_t(y)) * 0xC2B2AE3D27D4EB4Full) ^
               (uint64_t(uint32_t(z)) * 0x165667B19E3779F9ull);
    }
    glm::ivec3 coord(const glm::vec3& p) const {
        return glm::ivec3(glm::floor(p / cell));
    }
    void insert(const glm::vec3& p, uint32_t i) {
        const glm::ivec3 c = coord(p);
        buckets[key(c.x, c.y, c.z)].push_back(i);
    }
    template <class F>
    void for_each_near(const glm::vec3& p, F&& f) const {
        const glm::ivec3 c = coord(p);
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    auto it = buckets.find(key(c.x + dx, c.y + dy, c.z + dz));
                    if (it == buckets.end()) continue;
                    for (uint32_t i : it->second) f(i);
                }
    }
};

} // namespace

std::vector<EmitterProxy> fit_emitter_proxies(const SurfelSet& set, float spacing) {
    std::vector<EmitterProxy> out;
    if (set.emissive.empty() || spacing <= 0.0f) return out;

    // LOD 0 only. The coarser levels resample the same surface, so fitting them
    // too would emit the same light three more times.
    const uint32_t lod0_end = set.lods[0].offset + set.lods[0].count;

    struct Pt {
        glm::vec3 pos, nrm, rad;
        uint32_t index;
    };
    std::vector<Pt> pts;
    pts.reserve(set.emissive.size());
    for (const EmissiveSurfel& e : set.emissive) {
        if (e.index < set.lods[0].offset || e.index >= lod0_end) continue;
        const PackedSurfel& ps = set.packed[e.index];
        pts.push_back({surfel_position(set, ps), surfel_normal(ps),
                       glm::vec3(e.r, e.g, e.b), e.index});
    }
    if (pts.size() < kMinMembers) return out;

    const float link = kLinkSpacings * spacing;
    HashGrid grid;
    grid.cell = link;
    for (uint32_t i = 0; i < pts.size(); ++i) grid.insert(pts[i].pos, i);

    // Connected components by flood fill.
    std::vector<int> comp(pts.size(), -1);
    std::vector<uint32_t> stack;
    int ncomp = 0;
    for (uint32_t seed = 0; seed < pts.size(); ++seed) {
        if (comp[seed] >= 0) continue;
        const int id = ncomp++;
        comp[seed] = id;
        stack.clear();
        stack.push_back(seed);
        while (!stack.empty()) {
            const uint32_t i = stack.back();
            stack.pop_back();
            const Pt& a = pts[i];
            grid.for_each_near(a.pos, [&](uint32_t j) {
                if (comp[j] >= 0) return;
                const Pt& b = pts[j];
                if (glm::dot(a.nrm, b.nrm) < kLinkNormalDot) return;
                if (glm::distance(a.pos, b.pos) > link) return;
                comp[j] = id;
                stack.push_back(j);
            });
        }
    }

    // Sized with resize(), not with a constructor argument: written as
    // `groups(size_t(ncomp))` this parses as a function declaration.
    std::vector<std::vector<uint32_t>> groups;
    groups.resize(size_t(ncomp));
    for (uint32_t i = 0; i < pts.size(); ++i) groups[size_t(comp[i])].push_back(i);

    const float per_surfel_area = spacing * spacing;

    for (const std::vector<uint32_t>& g : groups) {
        if (g.size() < kMinMembers) continue;

        glm::vec3 center(0.0f), nsum(0.0f), rsum(0.0f);
        for (uint32_t i : g) {
            center += pts[i].pos;
            nsum += pts[i].nrm;
            rsum += pts[i].rad;
        }
        const float inv = 1.0f / float(g.size());
        center *= inv;
        rsum *= inv;
        if (glm::length(nsum) < 1e-8f) continue;
        const glm::vec3 normal = glm::normalize(nsum);

        // Tangent basis, then PCA in the plane: the 2x2 covariance's principal
        // axis has a closed form, so no iterative solver is needed.
        const glm::vec3 up = std::abs(normal.y) < 0.999f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
        const glm::vec3 t = glm::normalize(glm::cross(up, normal));
        const glm::vec3 b = glm::cross(normal, t);

        double sxx = 0.0, syy = 0.0, sxy = 0.0;
        for (uint32_t i : g) {
            const glm::vec3 d = pts[i].pos - center;
            const double x = glm::dot(d, t), y = glm::dot(d, b);
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
        }
        const double theta = 0.5 * std::atan2(2.0 * sxy, sxx - syy);
        const glm::vec3 e1 = t * float(std::cos(theta)) + b * float(std::sin(theta));
        const glm::vec3 e2 = glm::cross(normal, e1);

        float hu = 0.0f, hv = 0.0f;
        for (uint32_t i : g) {
            const glm::vec3 d = pts[i].pos - center;
            hu = std::max(hu, std::abs(glm::dot(d, e1)));
            hv = std::max(hv, std::abs(glm::dot(d, e2)));
        }
        // Half a spacing of padding: the extreme surfel is a sample point, not
        // the edge of the emitting surface.
        hu += 0.5f * spacing;
        hv += 0.5f * spacing;
        const float rect_area = 4.0f * hu * hv;
        if (rect_area <= 1e-9f) continue;

        const float covered = float(g.size()) * per_surfel_area;
        const float coverage = covered / rect_area;
        if (coverage < kMinCoverage) continue;

        EmitterProxy p;
        p.center = center;
        p.half_u = e1 * hu;
        p.half_v = e2 * hv;
        p.normal = normal;
        // Scale by coverage so the RECTANGLE emits the same power the island
        // did. Without this an L-shaped island lights the scene as if it were
        // the full bounding rectangle.
        p.radiance = rsum * std::min(coverage, 1.0f);
        p.coverage = coverage;
        p.members.reserve(g.size());
        for (uint32_t i : g) p.members.push_back(pts[i].index);
        out.push_back(std::move(p));
    }

    return out;
}

void EmitterSet::build(SurfelLibrary& lib, const std::vector<Instance>& instances) {
    placed_.clear();
    gpu_.clear();
    fitted_ = 0;
    rejected_ = 0;

    proxied_.assign(lib.total_surfels(), 0u);

    // Fit once per library entry, then instance.
    std::vector<std::vector<EmitterProxy>> per_entry(lib.entries().size());
    for (size_t e = 0; e < lib.entries().size(); ++e) {
        const SurfelLibrary::Entry& entry = lib.entries()[e];
        if (entry.set.emissive.empty()) continue;
        per_entry[e] = fit_emitter_proxies(entry.set, entry.set.lods[0].spacing);
        for (const EmitterProxy& p : per_entry[e])
            for (uint32_t m : p.members) proxied_[entry.base + m] = 1u;
    }

    for (size_t i = 0; i < instances.size(); ++i) {
        const int e = lib.entry_for(instances[i]);
        if (e < 0 || per_entry[size_t(e)].empty()) continue;
        for (const EmitterProxy& p : per_entry[size_t(e)])
            placed_.push_back({int(i), p});
    }

    fitted_ = uint32_t(placed_.size());
    proxied_buf_.data(proxied_.empty() ? nullptr : proxied_.data(),
                      GLsizeiptr(std::max<size_t>(proxied_.size(), 1)));

    refresh(instances);

    for (size_t i = 0; i < gpu_.size(); ++i) {
        const GpuEmitter& g = gpu_[i];
        gllib::logf(gllib::LogLevel::info,
                    "  emitter %zu: c(%.3f %.3f %.3f) hu %.3f hv %.3f area %.4f "
                    "n(%.2f %.2f %.2f) L(%.2f %.2f %.2f)",
                    i, g.p0.x, g.p0.y, g.p0.z,
                    glm::length(glm::vec3(g.p1)), glm::length(glm::vec3(g.p2)), g.p0.w,
                    g.p3.x, g.p3.y, g.p3.z, g.p1.w, g.p2.w, g.p3.w);
    }
    gllib::logf(gllib::LogLevel::info,
                "emitter proxies: %u placed from %zu sets, %u surfels proxied",
                fitted_, lib.entries().size(),
                uint32_t(std::count(proxied_.begin(), proxied_.end(), uint8_t(1))));
}

void EmitterSet::refresh(const std::vector<Instance>& instances) {
    gpu_.clear();
    gpu_.reserve(placed_.size());
    for (const Placed& pl : placed_) {
        if (pl.instance < 0 || size_t(pl.instance) >= instances.size()) continue;
        const glm::mat4& m = instances[size_t(pl.instance)].xform;
        const glm::mat3 lin(m);

        const glm::vec3 c = glm::vec3(m * glm::vec4(pl.proxy.center, 1.0f));
        glm::vec3 u = lin * pl.proxy.half_u;
        glm::vec3 v = lin * pl.proxy.half_v;
        glm::vec3 n = glm::cross(u, v);
        const float nl = glm::length(n);
        if (nl < 1e-12f) continue;
        n /= nl;

        // WINDING IS LOAD-BEARING, not bookkeeping.
        //
        // The polygon form factor is a signed sum: reverse the vertex order and
        // it comes out negative, and the shader's clamp then turns the light
        // off. cross(u, v) can land either way depending on how the PCA axes
        // fell out and on whether the instance transform is handed, so pin it
        // here -- swap the axes rather than negating one, which would also flip
        // the rectangle. The shader then walks the corners in the order that
        // winds counter-clockwise as seen from the emitting side.
        const glm::vec3 fitted = glm::normalize(lin * pl.proxy.normal);
        if (glm::dot(n, fitted) < 0.0f) {
            std::swap(u, v);
            n = -n;
        }

        GpuEmitter g;
        g.p0 = glm::vec4(c, 4.0f * nl);       // |u x v| * 4 is the world-space area
        g.p1 = glm::vec4(u, pl.proxy.radiance.r);
        g.p2 = glm::vec4(v, pl.proxy.radiance.g);
        g.p3 = glm::vec4(n, pl.proxy.radiance.b);
        gpu_.push_back(g);
    }

    // Always keep at least one element allocated: binding a zero-sized SSBO is
    // legal but every consumer then has to special-case the bind.
    if (gpu_.empty()) {
        buffer_.data(nullptr, GLsizeiptr(sizeof(GpuEmitter)));
    } else {
        buffer_.data(gpu_.data(), GLsizeiptr(gpu_.size() * sizeof(GpuEmitter)));
    }
}

} // namespace mosaic

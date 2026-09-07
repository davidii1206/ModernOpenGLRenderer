#include "direct.hpp"

#include <gllib/log.hpp>

#include <algorithm>
#include <cmath>

namespace sgi {
namespace {

// Two emissive triangles belong to the same island when they lie in the same
// plane. Cornell's panel is two triangles of one quad, so this is enough; a
// scene with several coplanar but disjoint panels would additionally need a
// connectivity pass, which is noted rather than written because nothing here
// exercises it.
bool same_plane(const Tri& a, const Tri& b) {
    if (glm::dot(a.n, b.n) < 0.999f) return false;
    return std::abs(glm::dot(b.p[0] - a.p[0], a.n)) < 1e-3f;
}

} // namespace

void EmitterSet::build(const std::vector<Tri>& tris) {
    emitters_.clear();
    source_area_ = proxy_area_ = 0.0;
    worst_plane_dev_ = 0.0f;

    std::vector<uint32_t> emissive;
    for (uint32_t i = 0; i < tris.size(); ++i)
        if (glm::dot(tris[i].emission, tris[i].emission) > 0.0f) emissive.push_back(i);

    std::vector<bool> used(emissive.size(), false);
    for (std::size_t a = 0; a < emissive.size(); ++a) {
        if (used[a]) continue;
        std::vector<uint32_t> island{emissive[a]};
        used[a] = true;
        for (std::size_t b = a + 1; b < emissive.size(); ++b)
            if (!used[b] && same_plane(tris[emissive[a]], tris[emissive[b]])) {
                island.push_back(emissive[b]);
                used[b] = true;
            }

        // Area-weighted centroid, normal and radiance over the island.
        glm::dvec3 c(0.0), rad(0.0);
        double area = 0.0;
        const glm::vec3 n = tris[island[0]].n;
        for (uint32_t t : island) {
            const Tri& tr = tris[t];
            const glm::dvec3 tc = (glm::dvec3(tr.p[0]) + glm::dvec3(tr.p[1]) +
                                   glm::dvec3(tr.p[2])) / 3.0;
            c += tc * double(tr.area);
            rad += glm::dvec3(tr.emission) * double(tr.area);
            area += double(tr.area);
        }
        if (area <= 0.0) continue;
        c /= area;
        rad /= area;
        source_area_ += area;

        // In-plane axes by MINIMUM-AREA BOUNDING RECTANGLE, not PCA.
        //
        // PCA is the obvious choice and it is wrong here for a reason that is
        // easy to miss: a quad built from two triangles lists its diagonal
        // corners TWICE, so the covariance is skewed toward that diagonal and the
        // principal axis comes out at 45 degrees. The bounding box in those axes
        // is then 1.86x the panel's real area -- measured, and caught by the area
        // assertion below rather than by looking at the image.
        //
        // Minimum-area is immune to vertex multiplicity, and it is exact for a
        // rectangular source, which is the case that matters. By the rotating
        // calipers result the optimal rectangle has a side collinear with a hull
        // edge, so testing every distinct edge direction of the point set finds
        // it; the counts here are tiny (4 distinct corners for Cornell).
        glm::vec3 t0, t1;
        {
            const glm::vec3 up = std::abs(n.y) < 0.9f ? glm::vec3(0, 1, 0)
                                                      : glm::vec3(1, 0, 0);
            t0 = glm::normalize(glm::cross(up, n));
            t1 = glm::cross(n, t0);
        }
        std::vector<glm::vec2> pts;
        for (uint32_t t : island)
            for (int k = 0; k < 3; ++k) {
                const glm::vec3 d = tris[t].p[k] - glm::vec3(c);
                worst_plane_dev_ = std::max(worst_plane_dev_, std::abs(glm::dot(d, n)));
                const glm::vec2 q(glm::dot(d, t0), glm::dot(d, t1));
                bool dup = false;
                for (const glm::vec2& e : pts)
                    if (glm::length(e - q) < 1e-5f) { dup = true; break; }
                if (!dup) pts.push_back(q);
            }

        double best_area = 1e30, best_ang = 0.0;
        for (std::size_t i = 0; i < pts.size(); ++i)
            for (std::size_t j = i + 1; j < pts.size(); ++j) {
                const glm::vec2 e = pts[j] - pts[i];
                if (glm::length(e) < 1e-6f) continue;
                const double ang = std::atan2(double(e.y), double(e.x));
                const double ca = std::cos(ang), sa = std::sin(ang);
                float lo0 = 1e30f, hi0 = -1e30f, lo1 = 1e30f, hi1 = -1e30f;
                for (const glm::vec2& q : pts) {
                    const float a0 = float(q.x * ca + q.y * sa);
                    const float a1 = float(-q.x * sa + q.y * ca);
                    lo0 = std::min(lo0, a0); hi0 = std::max(hi0, a0);
                    lo1 = std::min(lo1, a1); hi1 = std::max(hi1, a1);
                }
                const double area_r = double(hi0 - lo0) * double(hi1 - lo1);
                if (area_r < best_area) { best_area = area_r; best_ang = ang; }
            }

        const glm::vec3 u = t0 * float(std::cos(best_ang)) + t1 * float(std::sin(best_ang));
        const glm::vec3 v = glm::cross(n, u);

        // Extent along the chosen axes.
        float umin = 1e30f, umax = -1e30f, vmin = 1e30f, vmax = -1e30f;
        for (uint32_t t : island)
            for (int k = 0; k < 3; ++k) {
                const glm::vec3 d = tris[t].p[k] - glm::vec3(c);
                umin = std::min(umin, glm::dot(d, u)); umax = std::max(umax, glm::dot(d, u));
                vmin = std::min(vmin, glm::dot(d, v)); vmax = std::max(vmax, glm::dot(d, v));
            }
        const glm::vec3 centre = glm::vec3(c) + u * (0.5f * (umin + umax))
                                              + v * (0.5f * (vmin + vmax));
        const glm::vec3 half_u = u * (0.5f * (umax - umin));
        const glm::vec3 half_v = v * (0.5f * (vmax - vmin));
        const float rect_area = 4.0f * glm::length(half_u) * glm::length(half_v);
        proxy_area_ += double(rect_area);

        GpuEmitter e;
        e.p0 = glm::vec4(centre, rect_area);
        e.p1 = glm::vec4(half_u, float(rad.x));
        e.p2 = glm::vec4(half_v, float(rad.y));
        e.p3 = glm::vec4(n, float(rad.z));
        emitters_.push_back(e);
    }

    if (!emitters_.empty())
        b_.data(emitters_.data(), emitters_.size() * sizeof(GpuEmitter));

    const double ratio = source_area_ > 0.0 ? proxy_area_ / source_area_ : 0.0;
    gllib::logf(gllib::LogLevel::info,
                "emitters: %zu proxies, source area %.5f, proxy area %.5f "
                "(ratio %.4f), worst out-of-plane %.2e",
                emitters_.size(), source_area_, proxy_area_, ratio, worst_plane_dev_);
    // A rectangle fitted to a rectangular panel should reproduce its area. A
    // large ratio means the island is not rectangular and the proxy is a poor
    // stand-in -- which would show up only as a subtly wrong image, so say so.
    if (!emitters_.empty() && (ratio < 0.95 || ratio > 1.05))
        gllib::logf(gllib::LogLevel::warn,
                    "emitter proxy area is %.1f%% of the source area; the fit is "
                    "not rectangular and the direct term will be biased", ratio * 100.0);
    if (worst_plane_dev_ > 1e-3f)
        gllib::logf(gllib::LogLevel::warn,
                    "emitter island is not planar (%.2e out of plane)", worst_plane_dev_);
}

void EmitterSet::bind() const {
    if (!emitters_.empty()) b_.bind_base(kBindEmitters);
}

} // namespace sgi

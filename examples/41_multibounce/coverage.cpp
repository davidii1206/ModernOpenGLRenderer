#include "coverage.hpp"

#include <gllib/log.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mbg {

namespace {

// Cells are signed; bias them so the low-bit masks below work on a
// non-negative number and a surface straddling the origin is not a special case.
constexpr int64_t kBias = 1 << 20;

struct Probe {
    int64_t  cx, cy, cz;
    uint32_t cls;          // 0..5: +X -X +Y -Y +Z -Z
    uint32_t lod;
};

// The two axes of the class's tangent plane. The normal axis is deliberately
// ignored by the digit scheme (section 2.4), which is exactly where collisions
// come from on a surface that is not aligned with its class.
inline void tangent_axes(uint32_t cls, int& a, int& b) {
    const int n = int(cls >> 1);
    a = (n + 1) % 3;
    b = (n + 2) % 3;
}

inline int64_t axis(const Probe& p, int k) {
    return k == 0 ? p.cx : (k == 1 ? p.cy : p.cz);
}

} // namespace

bool run_coverage(const GBuffer& gb, const glm::mat4& inv_view_proj,
                  const glm::vec3& cam_pos, float scene_diagonal) {
    const int W = gb.width, H = gb.height;
    if (W <= 0 || H <= 0) return false;

    std::vector<float> depth(std::size_t(W) * H);
    std::vector<float> nrm(std::size_t(W) * H * 4);
    std::vector<uint8_t> alb(std::size_t(W) * H * 4);
    glGetTextureImage(gb.depth.handle(), 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                      GLsizei(depth.size() * sizeof(float)), depth.data());
    glGetTextureImage(gb.normal.handle(), 0, GL_RGBA, GL_FLOAT,
                      GLsizei(nrm.size() * sizeof(float)), nrm.data());
    glGetTextureImage(gb.albedo.handle(), 0, GL_RGBA, GL_UNSIGNED_BYTE,
                      GLsizei(alb.size()), alb.data());

    const int      block    = std::max(1, atoi(getenv("MBG_COV_BLOCK") ? getenv("MBG_COV_BLOCK") : "4"));
    const uint32_t cascades = uint32_t(std::max(2, atoi(getenv("MBG_COV_CASCADES") ? getenv("MBG_COV_CASCADES") : "5")));
    const bool     use_lod  = !getenv("MBG_COV_LOD") || atoi(getenv("MBG_COV_LOD")) != 0;
    const bool     dual     = !getenv("MBG_COV_DUAL") || atoi(getenv("MBG_COV_DUAL")) != 0;
    // Cascade-0 cell size. Swept, because coverage against it IS the experiment:
    // too large and every probe in a parent collapses to one digit tuple, too
    // small and the descendants that would fill the tuples do not exist.
    const float d0 = getenv("MBG_COV_D0") ? float(atof(getenv("MBG_COV_D0")))
                                          : scene_diagonal / 256.0f;
    // Where LOD 0 ends. Beyond it, spacing doubles per octave of Chebyshev
    // distance, which is what keeps probe footprint constant on screen.
    const float lod_ref = getenv("MBG_COV_LODREF") ? float(atof(getenv("MBG_COV_LODREF")))
                                                   : d0 * 32.0f;

    // cos(10 degrees from a class boundary): a normal whose two largest
    // components are within this ratio emits keys for BOTH classes, which is the
    // seam fix in section 2.3 and also the thing that inflates the probe count.
    const float dual_tol = 0.9f;

    std::vector<Probe> probes;
    probes.reserve(std::size_t(W / block) * (H / block));

    std::size_t blocks = 0, with_geometry = 0;
    for (int by = 0; by + block <= H; by += block) {
        for (int bx = 0; bx + block <= W; bx += block) {
            ++blocks;
            // Point-sampled at the block centre, never averaged: place.comp:14-19
            // has the argument, and it applies verbatim here.
            const int sx = std::min(bx + block / 2, W - 1);
            const int sy = std::min(by + block / 2, H - 1);
            const std::size_t px = std::size_t(sy) * W + sx;
            if (alb[px * 4 + 3] < 128) continue;       // background
            ++with_geometry;

            const float d = depth[px];
            const glm::vec2 uv((float(sx) + 0.5f) / float(W), (float(sy) + 0.5f) / float(H));
            glm::vec4 ndc(uv * 2.0f - 1.0f, d * 2.0f - 1.0f, 1.0f);
            glm::vec4 wp = inv_view_proj * ndc;
            const glm::vec3 x = glm::vec3(wp) / wp.w;

            glm::vec3 n(nrm[px * 4 + 0], nrm[px * 4 + 1], nrm[px * 4 + 2]);
            const float nl = glm::length(n);
            if (!(nl > 1e-6f)) continue;
            n /= nl;

            uint32_t lod = 0;
            if (use_lod) {
                const glm::vec3 r = glm::abs(x - cam_pos);
                const float cheb = std::max(r.x, std::max(r.y, r.z));
                if (cheb > lod_ref)
                    lod = uint32_t(std::min(15.0f, std::floor(std::log2(cheb / lod_ref))));
            }
            const float s = d0 * float(1u << lod);

            const float a[3] = {std::abs(n.x), std::abs(n.y), std::abs(n.z)};
            const int dom = (a[0] >= a[1] && a[0] >= a[2]) ? 0 : (a[1] >= a[2] ? 1 : 2);
            const int sec = (dom == 0) ? (a[1] >= a[2] ? 1 : 2)
                                       : (dom == 1 ? (a[0] >= a[2] ? 0 : 2)
                                                   : (a[0] >= a[1] ? 0 : 1));

            Probe p;
            p.cx  = int64_t(std::floor(x.x / s)) + kBias;
            p.cy  = int64_t(std::floor(x.y / s)) + kBias;
            p.cz  = int64_t(std::floor(x.z / s)) + kBias;
            p.lod = lod;
            const float sign_dom = (dom == 0 ? n.x : (dom == 1 ? n.y : n.z));
            p.cls = uint32_t(dom * 2 + (sign_dom < 0.0f ? 1 : 0));
            probes.push_back(p);

            // Near a class boundary, emit the second class too.
            if (dual && a[dom] > 0.0f && a[sec] / a[dom] > dual_tol) {
                const float sign_sec = (sec == 0 ? n.x : (sec == 1 ? n.y : n.z));
                p.cls = uint32_t(sec * 2 + (sign_sec < 0.0f ? 1 : 0));
                probes.push_back(p);
            }
        }
    }

    if (probes.empty()) {
        printf("[COV] no geometry in the G-buffer\n");
        return false;
    }

    // Unique c0 probes.
    auto key_of = [](const Probe& p, uint32_t shift) {
        const uint64_t cx = uint64_t(p.cx >> shift) & 0x3FFFFull;
        const uint64_t cy = uint64_t(p.cy >> shift) & 0x3FFFFull;
        const uint64_t cz = uint64_t(p.cz >> shift) & 0x3FFFFull;
        return (uint64_t(p.lod) << 57) | (uint64_t(p.cls) << 54) |
               (cx << 36) | (cy << 18) | cz;
    };

    std::unordered_map<uint64_t, Probe> c0;
    c0.reserve(probes.size() * 2);
    for (const Probe& p : probes) c0.emplace(key_of(p, 0), p);

    printf("[COV] %dx%d, block %d -> %zu blocks, %zu with geometry, "
           "%zu c0 probes (d0 %.4g, lod %s, dual-class %s)\n",
           W, H, block, blocks, with_geometry, c0.size(), double(d0),
           use_lod ? "on" : "off", dual ? "on" : "off");

    // For each cascade, group c0 probes by their ancestor and count the DISTINCT
    // tangent-plane digit tuples they occupy. That count over 4^n is the covered
    // fraction of every one of that parent's D0 bins.
    bool ok = true;
    printf("[COV] cascade  parents   mean cover   empty bins   worst parent\n");
    for (uint32_t n = 1; n < cascades; ++n) {
        const int64_t  mask  = (int64_t(1) << n) - 1;
        const double   slots = double(uint64_t(1) << (2 * n));   // 4^n

        std::unordered_map<uint64_t, std::unordered_set<uint64_t>> parent;
        parent.reserve(c0.size());
        for (const auto& kv : c0) {
            const Probe& p = kv.second;
            int ta, tb;
            tangent_axes(p.cls, ta, tb);
            const uint64_t tuple = uint64_t(axis(p, ta) & mask) |
                                   (uint64_t(axis(p, tb) & mask) << 32);
            parent[key_of(p, n)].insert(tuple);
        }

        double sum = 0.0, worst = 1.0;
        for (const auto& kv : parent) {
            const double c = std::min(1.0, double(kv.second.size()) / slots);
            sum += c;
            worst = std::min(worst, c);
        }
        const double mean = sum / double(parent.size());
        const double empty = 1.0 - mean;
        if (empty > 0.05) ok = false;
        printf("[COV] %7u  %7zu   %9.2f%%   %9.2f%%   %9.2f%%\n",
               n, parent.size(), mean * 100.0, empty * 100.0, worst * 100.0);
    }
    printf("[COV] gate: empty bins < 5%% on every cascade -> %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace mbg

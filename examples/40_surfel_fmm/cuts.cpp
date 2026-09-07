#include "cuts.hpp"

#include <gllib/log.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <map>
#include <unordered_map>

namespace sgi {
namespace {

constexpr float kSharpDot     = 0.6f;    // |n_i . n_j| below this is a sharp edge
constexpr float kCurvKappaMax = 200.0f;  // curvature clamp, 1/m
constexpr float kFlatKappa    = 0.5f;    // above this a face is "curved", no cuts
constexpr float kCutReachRadii = 4.5f;   // covers nee_occ up to the 4.0 slider top

// Quantised vertex key. Two triangles share an edge when their endpoints agree
// to 1e-4, which is what welds a glb's per-face duplicated vertices back into a
// topology without needing an index-level merge.
int64_t qkey(const glm::vec3& p) {
    return int64_t(std::llround(p.x * 10000.0f)) * 73856093 ^
           int64_t(std::llround(p.y * 10000.0f)) * 19349663 ^
           int64_t(std::llround(p.z * 10000.0f)) * 83492791;
}

struct EdgeAccum {
    int       count = 0;
    int       ti[2]{};
    glm::vec3 n[2]{};
    glm::vec3 ctr[2]{};
    glm::vec3 p[2][2]{};
    glm::vec3 opp[2]{};        // the third vertex, for face a's ownership triangle
};

} // namespace

FeatureEdges extract_feature_edges(const std::vector<Tri>& tris) {
    FeatureEdges out;
    out.tri_object.assign(tris.size(), 0);
    out.tri_curvature.assign(tris.size(), 0.0f);

    std::map<std::pair<int64_t, int64_t>, EdgeAccum> edge_map;
    for (std::size_t ti = 0; ti < tris.size(); ++ti) {
        const Tri& t = tris[ti];
        const glm::vec3 ctr = (t.p[0] + t.p[1] + t.p[2]) / 3.0f;
        for (int k = 0; k < 3; ++k) {
            const glm::vec3& p0 = t.p[k];
            const glm::vec3& p1 = t.p[(k + 1) % 3];
            int64_t ka = qkey(p0), kb = qkey(p1);
            if (ka > kb) std::swap(ka, kb);
            EdgeAccum& e = edge_map[{ka, kb}];
            if (e.count < 2) {
                e.ti[e.count]     = int(ti);
                e.n[e.count]      = t.n;
                e.ctr[e.count]    = ctr;
                e.p[e.count][0]   = p0;
                e.p[e.count][1]   = p1;
                e.opp[e.count]    = t.p[0] + t.p[1] + t.p[2] - p0 - p1;
            }
            ++e.count;
        }
    }

    // Objects: triangles joined by a shared edge are one object. Separate meshes
    // that merely touch stay separate, so a box standing on the floor is never
    // clipped by the floor's boundary edges and vice versa.
    {
        std::vector<int> parent(tris.size());
        for (std::size_t i = 0; i < parent.size(); ++i) parent[i] = int(i);
        auto find = [&](int x) {
            while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
            return x;
        };
        for (auto& [key, e] : edge_map) {
            if (e.count < 2) continue;
            const int a = find(e.ti[0]), b = find(e.ti[1]);
            if (a != b) parent[a] = b;
        }
        std::unordered_map<int, int> comp;
        int next = 0;
        for (std::size_t i = 0; i < tris.size(); ++i) {
            const int r = find(int(i));
            auto it = comp.find(r);
            if (it == comp.end()) it = comp.emplace(r, next++).first;
            out.tri_object[i] = it->second;
        }
    }

    // Curvature from SMOOTH dihedrals only: kappa = angle / edge length. Sharp
    // creases are excluded because both their faces are flat -- fidelity there
    // comes from the cut plane, not from calling the face curved and skipping it.
    for (auto& [key, e] : edge_map) {
        if (e.count != 2) continue;
        const float dn = std::abs(glm::dot(e.n[0], e.n[1]));
        if (dn < kSharpDot) continue;
        const float ang = std::acos(std::clamp(dn, 0.0f, 1.0f));
        const float len = std::max(glm::length(e.p[0][1] - e.p[0][0]), 1e-6f);
        const float kappa = std::min(ang / len, kCurvKappaMax);
        out.tri_curvature[std::size_t(e.ti[0])] =
            std::max(out.tri_curvature[std::size_t(e.ti[0])], kappa);
        out.tri_curvature[std::size_t(e.ti[1])] =
            std::max(out.tri_curvature[std::size_t(e.ti[1])], kappa);
    }

    for (auto& [key, e] : edge_map) {
        FeatEdge f;
        f.mid = 0.5f * (e.p[0][0] + e.p[0][1]);
        const glm::vec3 ev = e.p[0][1] - e.p[0][0];
        const float el = glm::length(ev);
        if (el < 1e-9f) continue;
        f.dir = ev / el;
        f.half_len = 0.5f * el;

        if (e.count == 1) {
            f.boundary = true;
            f.na = e.n[0];
            f.ctr_a = e.ctr[0];
            f.tri_a[0] = e.p[0][0];
            f.tri_a[1] = e.p[0][1];
            f.tri_a[2] = e.opp[0];
            f.obj_a = f.obj_b = out.tri_object[std::size_t(e.ti[0])];
            if (out.tri_curvature[std::size_t(e.ti[0])] < kFlatKappa)
                out.edges.push_back(f);
        } else if (e.count == 2 &&
                   std::abs(glm::dot(e.n[0], e.n[1])) < kSharpDot &&
                   out.tri_curvature[std::size_t(e.ti[0])] < kFlatKappa &&
                   out.tri_curvature[std::size_t(e.ti[1])] < kFlatKappa) {
            f.boundary = false;
            f.na = e.n[0];   f.nb = e.n[1];
            f.ctr_a = e.ctr[0]; f.ctr_b = e.ctr[1];
            f.obj_a = out.tri_object[std::size_t(e.ti[0])];
            f.obj_b = out.tri_object[std::size_t(e.ti[1])];
            out.edges.push_back(f);
        }
    }
    return out;
}

std::vector<glm::vec4> compute_surfel_cuts(const SurfelSet& set,
                                           const std::vector<uint32_t>& tri_of,
                                           const std::vector<Tri>& tris,
                                           const FeatureEdges& fe) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t N = set.count();
    std::vector<glm::vec4> cuts(std::size_t(N) * kSurfelCuts, glm::vec4(0.0f));
    if (N == 0 || fe.edges.empty()) return cuts;

    const auto& pr = set.pos_rad();
    uint32_t saturated = 0, with_cuts = 0;

    for (uint32_t i = 0; i < N; ++i) {
        const glm::vec3 ca = glm::vec3(pr[i]);
        const uint32_t  ti = tri_of[i];
        const glm::vec3 ni = tris[ti].n;
        const int       obj = fe.tri_object[ti];
        // Per-surfel reach: a disc can only cross an edge within its own
        // EFFECTIVE extent, which is the bake radius times the visibility
        // inflation (nee_occ). The cuts are baked once and nee_occ is a runtime
        // knob, so this covers the top of its slider rather than its default --
        // a surfel that gets a cut it never needed is free, one that needs a cut
        // it never got over-occludes at every silhouette.
        //
        // It stays PER SURFEL. Example 38 records what a global reach costs: a
        // plane through a distant crease slices unrelated parts of a curved
        // surface, and on the bunny it beheaded surfels 4 cm from the edge that
        // produced the plane.
        const float reach = kCutReachRadii * pr[i].w;

        int ncuts = 0;
        auto add_cut = [&](const glm::vec3& n, const glm::vec3& anchor) {
            const float d = -glm::dot(anchor, n);
            // A cut must never reject its own surfel. If the plane slices
            // through the centre this is not a clean crease for this disc but a
            // shallow surface intersection, and clipping there beheads it.
            // Border samples sit ON their plane (|dot| ~ 1e-6) and pass.
            if (glm::dot(ca, n) + d > 1e-4f) return;
            for (int k = 0; k < ncuts; ++k) {
                const glm::vec4& ex = cuts[std::size_t(i) * kSurfelCuts + k];
                if (glm::dot(glm::vec3(ex), n) > 0.98f && std::abs(ex.w - d) < 1e-3f)
                    return;
            }
            if (ncuts >= kSurfelCuts) { ++saturated; return; }
            cuts[std::size_t(i) * kSurfelCuts + ncuts] = glm::vec4(n, d);
            ++ncuts;
        };

        for (const FeatEdge& e : fe.edges) {
            if (ncuts >= kSurfelCuts) break;
            const float t = glm::clamp(glm::dot(ca - e.mid, e.dir), -e.half_len, e.half_len);
            const glm::vec3 cp = e.mid + e.dir * t;
            if (glm::length(ca - cp) > reach) continue;

            if (!e.boundary) {
                if (e.obj_a != obj && e.obj_b != obj) continue;
                // Which incident face is this surfel's own? Normal agreement,
                // exact for flat faces (1 against ~0).
                const float da = glm::dot(ni, e.na), db = glm::dot(ni, e.nb);
                if (std::max(da, db) < 0.5f) continue;      // a third surface passing by
                const bool own_is_a = da >= db;
                const glm::vec3& other   = own_is_a ? e.nb : e.na;
                const glm::vec3& own_ctr = own_is_a ? e.ctr_a : e.ctr_b;
                // Concavity from the OWN face's centroid, which is strictly
                // inside the face, so the test never degenerates for a surfel
                // sitting exactly on the edge. Concave means the own face
                // extends into the other's half-space, so keep that side.
                const bool concave = glm::dot(other, own_ctr - cp) > 0.0f;
                // The plane passes exactly through the edge -- no offset math,
                // so there is no float noise to flip a border sample's side.
                add_cut(concave ? -other : other, cp);
            } else {
                if (e.obj_a != obj) continue;
                // Ownership: the surfel must lie on the incident triangle's
                // plane and inside it. A box standing on the floor must not be
                // clipped by the floor's own boundary, and vice versa.
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
                if (bu < -1e-3f || bv < -1e-3f || 1.0f - bu - bv < -1e-3f) continue;
                glm::vec3 u = glm::cross(e.dir, e.na);
                const float ul = glm::length(u);
                if (ul < 1e-8f) continue;
                u /= ul;
                if (glm::dot(u, e.ctr_a - cp) < 0.0f) u = -u;   // edge -> interior
                add_cut(-u, cp);
            }
        }
        if (ncuts > 0) ++with_cuts;
    }

    const double ms = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0).count() * 1000.0;
    gllib::logf(gllib::LogLevel::info,
                "cuts: %zu feature edges, %u/%u surfels clipped (%.1f%%), "
                "%u saturated at %d planes, %.1f KB, %.2f ms",
                fe.edges.size(), with_cuts, N,
                N ? 100.0 * double(with_cuts) / double(N) : 0.0,
                saturated, kSurfelCuts,
                double(cuts.size() * sizeof(glm::vec4)) / 1024.0, ms);
    if (saturated > 0)
        gllib::logf(gllib::LogLevel::warn,
                    "%u surfels wanted more than %d cut planes; raise kSurfelCuts",
                    saturated, kSurfelCuts);
    return cuts;
}

void CutSet::build(const SurfelSet& set, const std::vector<Tri>& tris) {
    count_ = 0;
    if (set.count() == 0 || set.tri_of().size() != set.count()) {
        // No source mesh behind this set (the analytic gates build theirs
        // explicitly). One zeroed slot keeps the binding legal.
        const glm::vec4 zero(0.0f);
        b_.data(&zero, sizeof(zero));
        return;
    }
    const FeatureEdges fe = extract_feature_edges(tris);
    const std::vector<glm::vec4> cuts =
        compute_surfel_cuts(set, set.tri_of(), tris, fe);
    b_.data(cuts.data(), cuts.size() * sizeof(glm::vec4));
    count_ = set.count();
}

void CutSet::bind() const { b_.bind_base(kBindCuts); }

std::size_t CutSet::bytes() const {
    return std::size_t(count_) * kSurfelCuts * sizeof(glm::vec4);
}

} // namespace sgi

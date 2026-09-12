#include "validate.hpp"

#include <gllib/log.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <cstdlib>

namespace mbg {

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Report {
    int passed = 0, failed = 0;

    void check(int idx, const char* group, const char* what, double got, double expect,
               double tol, bool absolute = false) {
        const double err = std::abs(got - expect);
        const double rel = absolute ? err
                                    : err / std::max(1e-12, std::abs(expect));
        const bool ok = rel <= tol;
        ok ? ++passed : ++failed;
        printf("[GATE] %d %-7s %-28s got %-14.8g expect %-14.8g %s %-10.4g tol %-9.4g %s\n",
               idx, group, what, got, expect, absolute ? "abs" : "rel", rel, tol,
               ok ? "PASS" : "FAIL");
    }
};

bool wants(const std::string& names, const char* g) {
    if (names == "all") return true;
    return names.find(g) != std::string::npos;
}

// --- Test scenes ------------------------------------------------------------

// An axis-aligned box of half-extent h with its normals pointing INWARD, so a
// camera inside it is sealed: every direction of every hemisphere lands on a
// face. That is what makes the analytic answers below exact rather than
// approximate -- there is no horizon, no silhouette and no partial texel
// anywhere in the image.
std::vector<Tri> make_box(float h, glm::vec3 albedo, glm::vec3 emission) {
    const glm::vec3 c[8] = {
        {-h, -h, -h}, { h, -h, -h}, { h,  h, -h}, {-h,  h, -h},
        {-h, -h,  h}, { h, -h,  h}, { h,  h,  h}, {-h,  h,  h},
    };
    const int faces[6][4] = {
        {0, 1, 2, 3},   // -Z
        {5, 4, 7, 6},   // +Z
        {4, 0, 3, 7},   // -X
        {1, 5, 6, 2},   // +X
        {4, 5, 1, 0},   // -Y
        {3, 2, 6, 7},   // +Y
    };
    std::vector<Tri> out;
    for (const auto& f : faces) {
        const glm::vec3 e1 = c[f[1]] - c[f[0]];
        const glm::vec3 e2 = c[f[3]] - c[f[0]];
        glm::vec3 n = glm::normalize(glm::cross(e1, e2));
        // Point it at the centre, whatever the winding produced.
        if (glm::dot(n, -c[f[0]]) < 0.0f) n = -n;
        const int idx[2][3] = {{0, 1, 2}, {0, 2, 3}};
        for (const auto& t3 : idx) {
            Tri t;
            for (int k = 0; k < 3; ++k) t.p[k] = c[f[t3[k]]];
            t.n = n;
            t.albedo = albedo;
            t.emission = emission;
            t.area = 0.5f * glm::length(glm::cross(t.p[1] - t.p[0], t.p[2] - t.p[0]));
            out.push_back(t);
        }
    }
    return out;
}

// A rectangle 2a x 2b in the plane z = h, facing down at a camera at the origin.
void add_rect(std::vector<Tri>& out, float a, float b, float h, glm::vec3 albedo,
              glm::vec3 emission) {
    const glm::vec3 p[4] = {{-a, -b, h}, {a, -b, h}, {a, b, h}, {-a, b, h}};
    const int idx[2][3] = {{0, 1, 2}, {0, 2, 3}};
    for (const auto& t3 : idx) {
        Tri t;
        for (int k = 0; k < 3; ++k) t.p[k] = p[t3[k]];
        t.n = glm::vec3(0.0f, 0.0f, -1.0f);
        t.albedo = albedo;
        t.emission = emission;
        t.area = 0.5f * glm::length(glm::cross(t.p[1] - t.p[0], t.p[2] - t.p[0]));
        out.push_back(t);
    }
}

// Irradiance at a point directly below the centre of a parallel rectangle of a
// Lambertian emitter of radiance L. The configuration factor is the classic
// corner form quadrupled; E = PI * L * F because a fully enclosing hemisphere of
// radiance L gives PI*L at F = 1.
double rect_irradiance(double a, double b, double h, double L) {
    const double X = a / h, Y = b / h;
    const double sx = std::sqrt(1.0 + X * X), sy = std::sqrt(1.0 + Y * Y);
    const double f_corner = (1.0 / (2.0 * kPi)) *
                            (X / sx * std::atan(Y / sx) + Y / sy * std::atan(X / sy));
    return kPi * L * 4.0 * f_corner;
}

// The CPU mirror of mbg_onb in shaders/common/scene.glsl. Duff et al. 2017.
void onb(const glm::vec3& n, glm::vec3& t, glm::vec3& b) {
    const float s = n.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -1.0f / (s + n.z);
    const float c = n.x * n.y * a;
    t = glm::vec3(1.0f + s * n.x * n.x * a, s * c, -s * n.x);
    b = glm::vec3(c, s + n.y * n.y * a, -n.y);
}

// Moller-Trumbore, two-sided: the rasterizer does not cull either.
bool ray_tri(const glm::vec3& o, const glm::vec3& d, const Tri& tr, float& t_out) {
    const glm::vec3 e1 = tr.p[1] - tr.p[0];
    const glm::vec3 e2 = tr.p[2] - tr.p[0];
    const glm::vec3 pv = glm::cross(d, e2);
    const float det = glm::dot(e1, pv);
    if (std::abs(det) < 1e-20f) return false;
    const float inv = 1.0f / det;
    const glm::vec3 tv = o - tr.p[0];
    const float u = glm::dot(tv, pv) * inv;
    if (u < 0.0f || u > 1.0f) return false;
    const glm::vec3 qv = glm::cross(tv, e1);
    const float v = glm::dot(d, qv) * inv;
    if (v < 0.0f || u + v > 1.0f) return false;
    const float t = glm::dot(e2, qv) * inv;
    if (t <= 0.0f) return false;
    t_out = t;
    return true;
}

// Solve one point set and return the mean irradiance channel.
double mean_channel(const std::vector<glm::vec4>& e, int ch) {
    double s = 0.0;
    for (const glm::vec4& v : e) s += double(v[ch]);
    return e.empty() ? 0.0 : s / double(e.size());
}

} // namespace

bool run_gates(const std::string& names, Solver& solver, const SolveConfig& base,
               const Scene& scene, const std::vector<Tri>& tris) {
    Report r;
    printf("[GATE] running: %s\n", names.c_str());

    // --- 1. The quadrature table --------------------------------------------
    //
    // Before normalization the table must already integrate the hemisphere. If
    // it does not, normalizing to PI hides a shape error behind a correct total
    // and every subsequent number is quietly wrong.
    if (wants(names, "quad")) {
        for (uint32_t res : {8u, 16u, 32u}) {
            Quadrature q;
            q.build(res);
            char l1[64], l2[64];
            snprintf(l1, sizeof l1, "sum(dOmega) %ux%u", res, res);
            snprintf(l2, sizeof l2, "sum(cos dOmega) %ux%u", res, res);
            r.check(1, "quad", l1, q.sum_omega, 2.0 * kPi, 1e-5);
            r.check(1, "quad", l2, q.sum_cos, kPi, 1e-5);
        }
    }

    // --- 2. Sealed inside an emitter ----------------------------------------
    //
    // The end-to-end test of raster coverage plus quadrature: no gaps (a crack
    // reads low), no double coverage (atomicMin makes it harmless), correct
    // weights. Several normals, because a bug in the tangent frame or the fold
    // clipping shows on some orientations and not others.
    if (wants(names, "closed")) {
        const float L = 0.5f;
        Scene s;
        s.build(make_box(1.0f, glm::vec3(0.0f), glm::vec3(L)));
        SolveConfig cfg = base;
        cfg.bounces = 1;
        cfg.bias = 1e-4f;
        cfg.sky = glm::vec3(0.0f);

        std::vector<glm::vec4> pos, nrm;
        const glm::vec3 dirs[5] = {
            {0, 1, 0}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1},
            glm::normalize(glm::vec3(0.3f, 0.7f, -0.5f)),
        };
        for (const glm::vec3& d : dirs) {
            pos.push_back(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
            nrm.push_back(glm::vec4(d, 0.0f));
        }
        for (uint32_t res : {8u, 32u}) {
            cfg.res[0] = res;
            const std::vector<glm::vec4> e = solver.solve_points(s, cfg, pos, nrm);
            char lbl[64];
            snprintf(lbl, sizeof lbl, "E inside emitter %ux%u", res, res);
            r.check(2, "closed", lbl, mean_channel(e, 0), kPi * double(L), 1e-4);
            // Every orientation individually, not just the mean: opposite errors
            // would cancel.
            double worst = 0.0;
            for (const glm::vec4& v : e)
                worst = std::max(worst, std::abs(double(v.x) - kPi * double(L)));
            snprintf(lbl, sizeof lbl, "worst orientation %ux%u", res, res);
            r.check(2, "closed", lbl, worst, 0.0, 1e-4 * kPi * double(L), true);
        }
    }

    // --- 3. Analytic form factor --------------------------------------------
    //
    // The first test with a silhouette in it, so the first whose error is the
    // quadrature's own: coverage is binary per texel, so the answer converges
    // with resolution rather than being exact. Both resolutions are reported so
    // the convergence is visible instead of asserted blind.
    if (wants(names, "rect")) {
        const double L = 1.0, a = 0.5, b = 0.5, h = 1.0;
        std::vector<Tri> t;
        add_rect(t, float(a), float(b), float(h), glm::vec3(0.0f), glm::vec3(float(L)));
        Scene s;
        s.build(t);
        SolveConfig cfg = base;
        cfg.bounces = 1;
        cfg.bias = 1e-4f;
        cfg.sky = glm::vec3(0.0f);

        const std::vector<glm::vec4> pos{{0.0f, 0.0f, 0.0f, 1.0f}};
        const std::vector<glm::vec4> nrm{{0.0f, 0.0f, 1.0f, 0.0f}};
        const double expect = rect_irradiance(a, b, h, L);
        // Measured: 0.200 at 8x8, 0.043 at 16x16, 0.011 at 32x32. Coverage is
        // binary per texel, so the error lives entirely on the rectangle's
        // silhouette and falls off with resolution -- the tolerances below are
        // that sequence with room, and the shrinking assertion is the one that
        // would actually catch a regression, since a constant-factor bug would
        // hold the error flat while still passing a loose per-resolution bound.
        double prev = 1e30;
        bool shrinking = true;
        const double tol[3] = {0.25, 0.06, 0.02};
        int ti = 0;
        for (uint32_t res : {8u, 16u, 32u}) {
            cfg.res[0] = res;
            const std::vector<glm::vec4> e = solver.solve_points(s, cfg, pos, nrm);
            char lbl[64];
            snprintf(lbl, sizeof lbl, "E under rectangle %ux%u", res, res);
            r.check(3, "rect", lbl, double(e[0].x), expect, tol[ti++]);
            const double err = std::abs(double(e[0].x) - expect);
            if (err > prev * 0.5) shrinking = false;
            prev = err;
        }
        r.check(3, "rect", "error halves with resolution", shrinking ? 1.0 : 0.0, 1.0, 1e-9);
    }

    // --- 4. Occlusion --------------------------------------------------------
    //
    // The same rectangle with an opaque, non-emitting panel between it and the
    // camera. This is the property the whole technique exists for -- a rasterized
    // gather resolves visibility for free -- so it is worth its own assertion.
    if (wants(names, "occ")) {
        std::vector<Tri> t;
        add_rect(t, 0.5f, 0.5f, 1.0f, glm::vec3(0.0f), glm::vec3(1.0f));   // emitter
        add_rect(t, 0.9f, 0.9f, 0.5f, glm::vec3(0.5f), glm::vec3(0.0f));   // blocker
        Scene s;
        s.build(t);
        SolveConfig cfg = base;
        cfg.bounces = 1;
        cfg.bias = 1e-4f;
        cfg.sky = glm::vec3(0.0f);
        const std::vector<glm::vec4> pos{{0.0f, 0.0f, 0.0f, 1.0f}};
        const std::vector<glm::vec4> nrm{{0.0f, 0.0f, 1.0f, 0.0f}};
        cfg.res[0] = 32;
        const std::vector<glm::vec4> e = solver.solve_points(s, cfg, pos, nrm);
        r.check(4, "occ", "E behind an opaque panel", double(e[0].x), 0.0, 1e-6, true);
    }

    // --- 5. The rasterizer against a ray-cast oracle -------------------------
    //
    // Doc section 8.2 milestone 3, with a CPU ray cast standing in for the
    // hardware path: for every texel of every camera, the triangle the compute
    // rasterizer chose must be the triangle a ray through that texel's centre
    // hits first. This is the assertion that catches a fold-clipping bug, an
    // edge-function orientation bug or a depth-key packing bug -- none of which
    // is visible in an irradiance number, because they move visibility around
    // without changing how much of the hemisphere is covered.
    if (wants(names, "oracle")) {
        SolveConfig cfg = base;
        cfg.bounces = 1;
        cfg.res[0] = 32;

        // Cameras on the scene's own surfaces, which is where the real ones go.
        std::mt19937 rng(1234u);
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        std::vector<glm::vec4> pos, nrm;
        const uint32_t kCams = 64;
        for (uint32_t i = 0; i < kCams; ++i) {
            const Tri& t = tris[rng() % tris.size()];
            float a = u01(rng), b = u01(rng);
            if (a + b > 1.0f) { a = 1.0f - a; b = 1.0f - b; }
            pos.push_back(glm::vec4(t.p[0] + a * (t.p[1] - t.p[0]) + b * (t.p[2] - t.p[0]), 1.0f));
            nrm.push_back(glm::vec4(t.n, 0.0f));
        }

        Quadrature q;
        q.build(cfg.res[0]);
        const std::vector<uint32_t> vis = solver.raster_visibility(scene, cfg, pos, nrm);
        const uint32_t texels = cfg.res[0] * cfg.res[0];

        std::size_t mismatch = 0, empty_gpu = 0, empty_cpu = 0, total = 0;
        std::size_t interior = 0;
        double lost_weight = 0.0;
        const bool verbose = getenv("MBG_GATE_VERBOSE") != nullptr;
        for (uint32_t c = 0; c < kCams; ++c) {
            glm::vec3 N = glm::normalize(glm::vec3(nrm[c]));
            glm::vec3 T, B;
            onb(N, T, B);
            const glm::vec3 P = glm::vec3(pos[c]) + N * cfg.bias;
            for (uint32_t i = 0; i < texels; ++i) {
                const glm::vec3 dl = glm::vec3(q.texels[i]);
                const glm::vec3 d = T * dl.x + B * dl.y + N * dl.z;
                int best = -1;
                float bestt = 1e30f;
                for (std::size_t k = 0; k < tris.size(); ++k) {
                    float t;
                    if (ray_tri(P, d, tris[k], t) && t < bestt) { bestt = t; best = int(k); }
                }
                const uint32_t key = vis[c * texels + i];
                const bool gpu_empty = key == 0xFFFFFFFFu;
                ++total;
                // A texel on the outermost ring of the square looks along the
                // HORIZON: its direction has cos(theta) of order 1/res^2 and the
                // triangle covering it has been clipped at z = 0, so whether its
                // centre lands inside the projected polygon is decided at the
                // last float bit. Disagreement there is expected; disagreement
                // anywhere else is a bug. Both are reported, and the assertion
                // is on the energy the disagreement carries, which is the only
                // thing that can actually move an image.
                const uint32_t tx = i % cfg.res[0], ty = i / cfg.res[0];
                const bool rim = tx == 0 || ty == 0 || tx + 1 == cfg.res[0] ||
                                 ty + 1 == cfg.res[0];
                if (gpu_empty && best < 0) continue;
                if (gpu_empty) {
                    ++empty_gpu;
                    if (!rim) ++interior;
                    lost_weight += double(q.texels[i].w);
                    if (verbose)
                        printf("[GATE]   miss cam %u texel (%u,%u) rim=%d dir (%.6f %.6f %.6f) "
                               "diag=%d antidiag=%d cpu tri %d at %.6f\n",
                               c, i % cfg.res[0], i / cfg.res[0], int(rim),
                               dl.x, dl.y, dl.z,
                               int(i % cfg.res[0] == i / cfg.res[0]),
                               int(i % cfg.res[0] + i / cfg.res[0] + 1 == cfg.res[0]),
                               best, bestt);
                    continue;
                }
                if (best < 0) {
                    ++empty_cpu;
                    if (!rim) ++interior;
                    lost_weight += double(q.texels[i].w);
                    continue;
                }
                const uint32_t gt = key & 0xFFFFu;
                if (int(gt) == best) continue;
                // A different triangle at the same distance is a tie on a shared
                // edge or a coplanar seam, which the atomic is free to break
                // either way. Only a different SURFACE is a mismatch.
                float t2;
                if (ray_tri(P, d, tris[gt], t2) &&
                    std::abs(t2 - bestt) <= 1e-4f * std::max(1.0f, bestt))
                    continue;
                ++mismatch;
                if (!rim) ++interior;
                lost_weight += double(q.texels[i].w);
            }
        }
        printf("[GATE] 5 oracle  %zu texels over %u cameras: %zu wrong triangle, "
               "%zu GPU-only miss, %zu CPU-only miss, %zu of them off the horizon rim\n",
               total, kCams, mismatch, empty_gpu, empty_cpu, interior);
        r.check(5, "oracle", "wrong triangle fraction",
                double(mismatch) / double(std::max<std::size_t>(1, total)), 0.0, 1e-9, true);
        r.check(5, "oracle", "disagreements off the rim",
                double(interior), 0.0, 0.5, true);
        // Every disagreeing texel's own quadrature weight, as a fraction of the
        // hemisphere's PI. This is the bound on how much of the image the
        // rasterizer and the oracle could possibly disagree about.
        r.check(5, "oracle", "energy fraction in dispute", lost_weight / kPi, 0.0, 1e-4, true);
    }

    // --- 6. Multi-bounce energy ---------------------------------------------
    //
    // A closed enclosure with uniform emission L and uniform albedo p has a
    // uniform solution: L_out = L + p*L_out, so E = PI*L/(1-p) everywhere. N
    // camera levels truncate that series at N terms, which gives an exact
    // expected value per bounce count and makes this the one gate that scores
    // the RECURSION rather than a single gather -- the /PI in gather.comp, the
    // per-tile albedo mass from the spawn, and the deepest-level-first ordering
    // all have to be right for it to pass.
    if (wants(names, "series")) {
        const double L = 0.25, p = 0.5;
        Scene s;
        s.build(make_box(1.0f, glm::vec3(float(p)), glm::vec3(float(L))));
        SolveConfig cfg = base;
        cfg.bias = 1e-4f;
        cfg.sky = glm::vec3(0.0f);
        for (uint32_t i = 0; i < kMaxLevels; ++i) { cfg.res[i] = 16; cfg.block[i] = 4; }

        const std::vector<glm::vec4> pos{{0.0f, 0.0f, 0.0f, 1.0f},
                                         {0.4f, -0.3f, 0.2f, 1.0f}};
        const std::vector<glm::vec4> nrm{{0.0f, 1.0f, 0.0f, 0.0f},
                                         {0.0f, 0.0f, 1.0f, 0.0f}};
        for (uint32_t n = 1; n <= 3; ++n) {
            cfg.bounces = n;
            const std::vector<glm::vec4> e = solver.solve_points(s, cfg, pos, nrm);
            double expect = 0.0, pk = 1.0;
            for (uint32_t k = 0; k < n; ++k) { expect += pk; pk *= p; }
            expect *= kPi * L;
            char lbl[64];
            snprintf(lbl, sizeof lbl, "E after %u level%s", n, n == 1 ? "" : "s");
            r.check(6, "series", lbl, mean_channel(e, 0), expect, 2e-3);
        }
        // And the limit the series is converging to, for scale.
        printf("[GATE] 6 series  infinite-bounce value would be %.6f\n",
               kPi * L / (1.0 - p));
    }

    printf("[GATE] %d passed, %d failed\n", r.passed, r.failed);
    return r.failed == 0;
}

} // namespace mbg

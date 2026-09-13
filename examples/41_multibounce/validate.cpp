#include "validate.hpp"

#include <gllib/log.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdio>
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

// Lambert's formula on the CPU: the projected solid angle of a triangle seen
// from `pos` with normal `nrm`, clipped to that hemisphere. The mirror of
// mbg_emitter_unshadowed in raster.comp, and the reference the texeldir gate
// scores the shader against.
double cpu_unshadowed(const Tri& tr, const glm::vec3& pos, const glm::vec3& nrm,
                      float bias) {
    const double h = double(glm::dot(tr.n, pos - tr.p[0]));
    if (h < 0.0 && !tr.double_sided) return 0.0;
    // Coplanar means measure zero; merely passing through the receiver while
    // perpendicular to it does not. Both conditions, as in the shader.
    if (std::abs(h) < 2.0 * double(bias) &&
        std::abs(glm::dot(tr.n, nrm)) > 0.9f) return 0.0;

    glm::vec3 poly[4];
    int n = 0;
    for (int i = 0; i < 3; ++i) {
        const glm::vec3 a = tr.p[i] - pos;
        const glm::vec3 b = tr.p[(i + 1) % 3] - pos;
        const float da = glm::dot(nrm, a), db = glm::dot(nrm, b);
        if (da >= 0.0f && n < 4) poly[n++] = a;
        if ((da >= 0.0f) != (db >= 0.0f) && n < 4) poly[n++] = a + (b - a) * (da / (da - db));
    }
    if (n < 3) return 0.0;

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const glm::vec3 a = glm::normalize(poly[i]);
        const glm::vec3 b = glm::normalize(poly[(i + 1) % n]);
        const glm::vec3 x = glm::cross(a, b);
        const double len = double(glm::length(x));
        if (len < 1e-9) continue;
        sum += std::acos(std::clamp(double(glm::dot(a, b)), -1.0, 1.0)) *
               double(glm::dot(nrm, x / float(len)));
    }
    return 0.5 * std::abs(sum);
}

// Solve one point set and return the mean irradiance channel.
double mean_channel(const std::vector<glm::vec4>& e, int ch) {
    double s = 0.0;
    for (const glm::vec4& v : e) s += double(v[ch]);
    return e.empty() ? 0.0 : s / double(e.size());
}

} // namespace

bool run_gates(const std::string& names, Solver& solver, const SolveConfig& base_in,
               const Scene& scene, const std::vector<Tri>& tris) {
    Report r;
    printf("[GATE] running: %s\n", names.c_str());
    // The oracle and texeldir gates rebuild the receiver's frame on the CPU with
    // mbg_onb's exact arithmetic, so the shader must not rotate it underneath
    // them. Every gate runs unjittered for that reason; the rotation changes
    // which directions are sampled, never how much energy the set carries, which
    // is what these assertions are about.
    SolveConfig base = base_in;
    base.jitter = false;
    // Every assertion below is about the emitters and the geometry, and every
    // one of them predicts an answer on the CPU from those alone. An environment
    // adds energy none of those predictions account for, so it is cleared here
    // rather than in each gate -- MBG_SKY=0.05 MBG_GATE=all used to be a way to
    // fail a correct renderer.
    base.sky = SkyLight{};
    // And on the DETERMINISTIC estimator. Every assertion below predicts an
    // exact number from the geometry, which a stochastic estimator only matches
    // in expectation -- so the branching tile recursion is what the 31 gates
    // measure, and the path estimator is measured against it (the `paths` gate
    // at the end, which is the one that may legitimately be noisy).
    base.paths = 0;

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
        cfg.sky = SkyLight{};

        std::vector<glm::vec4> pos, nrm;
        const glm::vec3 dirs[5] = {
            {0, 1, 0}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1},
            glm::normalize(glm::vec3(0.3f, 0.7f, -0.5f)),
        };
        for (const glm::vec3& d : dirs) {
            pos.push_back(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
            nrm.push_back(glm::vec4(d, 0.0f));
        }
        // Receivers ON the faces, not just in the middle. A point on a wall of a
        // sealed emitter still reads exactly PI*L -- its own face is coplanar and
        // contributes nothing, and the other five cover its hemisphere. This is
        // the configuration every HIT POINT in the recursion is in, so it is the
        // one that has to be right for the per-texel direct term to be right.
        const float inset = 0.999f;
        const glm::vec3 face_pos[6] = {
            {0, 0, -inset}, {0, 0, inset}, {-inset, 0, 0},
            {inset, 0, 0}, {0, -inset, 0}, {0, inset, 0}};
        const glm::vec3 face_nrm[6] = {
            {0, 0, 1}, {0, 0, -1}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
        for (int i = 0; i < 6; ++i) {
            pos.push_back(glm::vec4(face_pos[i], 1.0f));
            nrm.push_back(glm::vec4(face_nrm[i], 0.0f));
        }
        // And receivers close to an edge, where a neighbouring face is nearly
        // edge-on and the clip has the least room.
        pos.push_back(glm::vec4(0.98f, 0.0f, -inset, 1.0f));
        nrm.push_back(glm::vec4(0.0f, 0.0f, 1.0f, 0.0f));

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
        cfg.sky = SkyLight{};

        const std::vector<glm::vec4> pos{{0.0f, 0.0f, 0.0f, 1.0f}};
        const std::vector<glm::vec4> nrm{{0.0f, 0.0f, 1.0f, 0.0f}};
        const double expect = rect_irradiance(a, b, h, L);

        // Two estimators, two different properties to assert.
        //
        // With the analytic direct term on (the default), the magnitude is
        // Lambert's formula and the only sampled quantity is a visibility
        // fraction that is identically 1 here -- so the answer must be right AND
        // must not depend on the target resolution at all. The second half is
        // the stronger claim and the one that says the quadrature has been taken
        // out of the direct path.
        //
        // With it off, the emitter is found by the raster and coverage is binary
        // per texel, so the error lives on the silhouette and has to SHRINK with
        // resolution. Measured: 0.200 at 8x8, 0.043 at 16x16, 0.011 at 32x32. A
        // constant-factor bug would hold it flat while still passing any loose
        // per-resolution bound, which is why the assertion is on the shrinking.
        cfg.nee = true;
        double first = 0.0, spread = 0.0;
        for (uint32_t res : {8u, 16u, 32u}) {
            cfg.res[0] = res;
            const std::vector<glm::vec4> e = solver.solve_points(s, cfg, pos, nrm);
            char lbl[64];
            snprintf(lbl, sizeof lbl, "analytic, %ux%u target", res, res);
            r.check(3, "rect", lbl, double(e[0].x), expect, 1e-3);
            if (res == 8u) first = double(e[0].x);
            else spread = std::max(spread, std::abs(double(e[0].x) - first));
        }
        r.check(3, "rect", "analytic is resolution-free", spread, 0.0, 1e-6, true);

        cfg.nee = false;
        double prev = 1e30;
        bool shrinking = true;
        const double tol[3] = {0.25, 0.06, 0.02};
        int ti = 0;
        for (uint32_t res : {8u, 16u, 32u}) {
            cfg.res[0] = res;
            const std::vector<glm::vec4> e = solver.solve_points(s, cfg, pos, nrm);
            char lbl[64];
            snprintf(lbl, sizeof lbl, "quadrature, %ux%u target", res, res);
            r.check(3, "rect", lbl, double(e[0].x), expect, tol[ti++]);
            const double err = std::abs(double(e[0].x) - expect);
            if (err > prev * 0.5) shrinking = false;
            prev = err;
        }
        r.check(3, "rect", "quadrature error halves", shrinking ? 1.0 : 0.0, 1.0, 1e-9);
        cfg.nee = true;
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
        cfg.sky = SkyLight{};
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
                const uint32_t gt = key;
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

    // --- 5b. The per-texel direct term --------------------------------------
    //
    // The reconstruction evaluates the unshadowed direct irradiance at every
    // texel's HIT POINT, and that number is what a whole tile's mass gets
    // multiplied by. No camera-based gate reaches it: a camera gate checks the
    // formula at a point it chose itself, while this path has to choose the
    // point -- from a quantized visibility key, a texel direction and a plane
    // intersection. Getting the point wrong is invisible in every other
    // assertion and shows up only as a few percent of missing bounce energy.
    //
    // Inside a sealed emitter every hit point reads PI*L, so the cosine-weighted
    // average over the hemisphere must be PI*L too, whatever the resolution.
    if (wants(names, "texeldir")) {
        const double L = 0.25;
        Scene s;
        s.build(make_box(1.0f, glm::vec3(0.5f), glm::vec3(float(L))));
        SolveConfig cfg = base;
        cfg.bounces = 1;
        cfg.bias = 1e-4f;
        cfg.sky = SkyLight{};
        const std::vector<glm::vec4> pos{{0.0f, 0.0f, 0.0f, 1.0f}};
        const std::vector<glm::vec4> nrm{{0.0f, 1.0f, 0.0f, 0.0f}};

        for (uint32_t res : {8u, 16u, 32u}) {
            cfg.res[0] = res;
            Quadrature q;
            q.build(res);
            const std::vector<uint32_t> raw = solver.raster_visibility(s, cfg, pos, nrm, 2);

            // The same hemisphere, evaluated entirely on the CPU: ray cast to
            // find the hit, then Lambert at that hit. Any per-texel disagreement
            // is the shader choosing a different hit point, which is the one
            // failure mode this gate exists to catch.
            const std::vector<Tri> box = make_box(1.0f, glm::vec3(0.5f), glm::vec3(float(L)));
            glm::vec3 T, B;
            const glm::vec3 N = glm::vec3(nrm[0]);
            onb(N, T, B);
            const glm::vec3 P = glm::vec3(pos[0]) + N * cfg.bias;

            double acc = 0.0, acc_cpu = 0.0, worst = 0.0;
            int worst_i = -1, reported = 0;
            for (uint32_t i = 0; i < res * res; ++i) {
                float v;
                std::memcpy(&v, &raw[i], sizeof v);
                acc += double(v) * double(q.texels[i].w);

                const glm::vec3 dl = glm::vec3(q.texels[i]);
                const glm::vec3 d = T * dl.x + B * dl.y + N * dl.z;
                int best = -1;
                float bestt = 1e30f;
                for (std::size_t k = 0; k < box.size(); ++k) {
                    float t;
                    if (ray_tri(P, d, box[k], t) && t < bestt) { bestt = t; best = int(k); }
                }
                double cpu = 0.0;
                if (best >= 0) {
                    const glm::vec3 hp = P + d * bestt;
                    const glm::vec3 hn = glm::dot(box[best].n, d) < 0.0f ? box[best].n
                                                                        : -box[best].n;
                    for (const Tri& e : box)
                        cpu += cpu_unshadowed(e, hp, hn, cfg.bias) *
                               double(e.emission.x);
                }
                acc_cpu += cpu * double(q.texels[i].w);

                const double diff = std::abs(double(v) - cpu);
                if (diff > worst) { worst = diff; worst_i = int(i); }
                if (std::abs(cpu - kPi * L) > 1e-3 * kPi * L && reported < 6) {
                    ++reported;
                    const glm::vec3 hp = best >= 0 ? P + d * bestt : glm::vec3(0.0f);
                    printf("[GATE] 7 texeldir  texel %u (%u,%u) dir (%.3f %.3f %.3f) "
                           "hit tri %d at (%.4f %.4f %.4f) t=%.4f: GPU %.6f CPU %.6f\n",
                           i, i % res, i / res, dl.x, dl.y, dl.z, best,
                           hp.x, hp.y, hp.z, bestt, double(v), cpu);
                }
            }
            char lbl[64];
            snprintf(lbl, sizeof lbl, "mean E_dir at hits %ux%u", res, res);
            // The average is over PI steradians of cosine weight, so divide it out.
            r.check(7, "texeldir", lbl, acc / kPi, kPi * L, 2e-3);
            snprintf(lbl, sizeof lbl, "vs CPU hit points %ux%u", res, res);
            r.check(7, "texeldir", lbl, worst, 0.0, 1e-3 * kPi * L, true);
            (void)worst_i;
            (void)acc_cpu;
        }
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
        cfg.sky = SkyLight{};
        // Overridable so the gate can be bisected against resolution and tile
        // size when it fails -- which is how the coplanar-emitter bug and the
        // grazing-hit bug below were both localized.
        uint32_t gres = 16, gblock = 4;
        if (const char* v = getenv("MBG_GATE_RES")) gres = uint32_t(std::max(2, atoi(v)));
        if (const char* v = getenv("MBG_GATE_BLOCK")) gblock = uint32_t(std::max(1, atoi(v)));
        for (uint32_t i = 0; i < kMaxLevels; ++i) { cfg.res[i] = gres; cfg.block[i] = gblock; }

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

    // --- 8. The path estimator is unbiased, and so is the roulette -----------
    //
    // The same closed enclosure, and the same exact answer per bounce count --
    // but measured with the SINGLE-SAMPLE CONTINUATION estimator, which the 31
    // gates above deliberately do not exercise because they all assert exact
    // numbers and this one is stochastic.
    //
    // Stochastic is not the same as approximate, and that is the whole point of
    // the gate. An importance-sampled continuation divided by its own pdf, and a
    // roulette that divides survivors by their survival probability, are both
    // exactly unbiased: the expectation is the analytic series, and the only
    // thing the sample count buys is how tightly one run lands on it. So the
    // tolerance here is a STANDARD ERROR argument rather than a bound on
    // systematic error -- 1.5% over 512 receivers x 64 paths -- and a bug in
    // either weight shows up as a bias that no amount of averaging removes.
    //
    // Roulette is then asserted twice over: once at a depth where it cannot
    // fire, where it must change nothing at all, and once past it, where it must
    // still land on the same number while visibly terminating paths.
    if (wants(names, "paths")) {
        const double L = 0.25, p = 0.5;
        Scene s;
        s.build(make_box(1.0f, glm::vec3(float(p)), glm::vec3(float(L))));
        SolveConfig cfg = base;
        cfg.bias = 1e-4f;
        cfg.sky = SkyLight{};
        cfg.paths = 64;
        for (uint32_t i = 0; i < kMaxLevels; ++i) { cfg.res[i] = 16; cfg.block[i] = 4; }

        // Receivers spread over the floor and one wall, because the random
        // streams are hashed from POSITION: 512 distinct points are 512
        // independent draws, where 512 copies of one point would be one draw
        // repeated and would measure nothing.
        std::vector<glm::vec4> pos, nrm;
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j) {
                const float u = (float(i) + 0.5f) / 16.0f * 1.6f - 0.8f;
                const float v = (float(j) + 0.5f) / 16.0f * 1.6f - 0.8f;
                pos.push_back(glm::vec4(u, -1.0f, v, 1.0f));
                nrm.push_back(glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));
                pos.push_back(glm::vec4(-1.0f, u, v, 1.0f));
                nrm.push_back(glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
            }

        auto series = [&](uint32_t n) {
            double e = 0.0, pk = 1.0;
            for (uint32_t k = 0; k < n; ++k) { e += pk; pk *= p; }
            return e * kPi * L;
        };

        for (uint32_t n = 2; n <= kMaxLevels; ++n) {
            cfg.bounces = n;
            cfg.rr = 0.0f;
            const std::vector<glm::vec4> e = solver.solve_points(s, cfg, pos, nrm);
            char lbl[64];
            snprintf(lbl, sizeof lbl, "E after %u levels, %u paths", n, cfg.paths);
            r.check(8, "paths", lbl, mean_channel(e, 0), series(n), 1.5e-2);
        }

        // Roulette, where it cannot fire. Throughput after k bounces is about
        // p^k = 0.5^k, so at 3 levels the deepest path still carries 0.25 and a
        // threshold of 0.15 never triggers: the two runs must agree EXACTLY,
        // which also proves the roulette code path is not perturbing the stream
        // of random numbers the continuation draws from.
        cfg.bounces = kMaxLevels;
        cfg.rr = 0.0f;
        const double no_rr = mean_channel(solver.solve_points(s, cfg, pos, nrm), 0);
        cfg.rr = 0.15f;
        const double dormant = mean_channel(solver.solve_points(s, cfg, pos, nrm), 0);
        r.check(8, "paths", "roulette dormant above threshold", dormant, no_rr, 1e-9);

        // And where it does fire: a threshold above the throughput at every
        // depth kills most paths after the first bounce, and the answer must not
        // move. This is the assertion that the 1/q division is there and right;
        // without it the result comes back short by whatever fraction died.
        cfg.bounces = kMaxLevels;
        cfg.rr = 0.0f;
        const double full = mean_channel(solver.solve_points(s, cfg, pos, nrm), 0);
        cfg.rr = 0.9f;
        const double rolled = mean_channel(solver.solve_points(s, cfg, pos, nrm), 0);
        r.check(8, "paths", "roulette unbiased below threshold", rolled, full, 2.5e-2);
        printf("[GATE] 8 paths   roulette at 0.9: %.6f vs %.6f unrouletted "
               "(analytic %.6f)\n", rolled, full, series(kMaxLevels));
    }

    // --- 9. Reordering the traversal cannot change what it finds -------------
    //
    // The oracle above is the assertion that matters for the rasterizer, but it
    // cannot reach the code this one is about: it runs a CPU ray cast per texel
    // per triangle, so it is only affordable on Cornell -- and Cornell is 32
    // triangles, one cluster, which is below the threshold where the cooperative
    // traversal and the cluster levels switch on at all. Everything
    // mbg_order_groups and the 2x2 texel mapping do is invisible to it.
    //
    // Both of those are PURE REORDERINGS. Visiting the coarse groups
    // nearest-first changes which box is tested when; handing a thread a tile
    // instead of a stride changes which thread owns which texel. Neither changes
    // what any texel is compared against, and the distance bound they exist to
    // tighten is conservative, so it can only skip boxes that could not have
    // won. The visibility buffer must therefore come back BIT-IDENTICAL, and
    // that is a far stronger statement than an energy tolerance: a bound that
    // pruned one triangle too many shows up as a single changed texel here and
    // would hide inside any average.
    //
    // GPU-only, so it costs one extra solve per variant and runs on a scene of
    // any size -- which is the point, since it is only meaningful on one big
    // enough to turn the coop path on.
    if (wants(names, "order")) {
        SolveConfig cfg = base;
        cfg.bounces = 1;
        cfg.res[0] = 32;

        std::mt19937 rng(1234u);
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        std::vector<glm::vec4> pos, nrm;
        const uint32_t kCams = 256;
        for (uint32_t i = 0; i < kCams; ++i) {
            const Tri& t = tris[rng() % tris.size()];
            float a = u01(rng), b = u01(rng);
            if (a + b > 1.0f) { a = 1.0f - a; b = 1.0f - b; }
            pos.push_back(glm::vec4(t.p[0] + a * (t.p[1] - t.p[0]) + b * (t.p[2] - t.p[0]), 1.0f));
            nrm.push_back(glm::vec4(t.n, 0.0f));
        }

        // raster_visibility reads the buffer back, so it synchronises and the
        // wall clock around it is the hemisphere's cost and nothing else. This
        // is the same measurement the culling and cooperative-traversal commits
        // reported, which is why the timing lives in the gate rather than in a
        // bench frame: MBG_BENCH averages a pass over a frame that is dominated
        // by the per-pixel direct term.
        auto timed = [&](SolveConfig c, double& ms) {
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<uint32_t> v = solver.raster_visibility(scene, c, pos, nrm);
            ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0).count();
            return v;
        };

        Quadrature qo;
        qo.build(cfg.res[0]);
        const uint32_t texels_o = cfg.res[0] * cfg.res[0];

        SolveConfig ref = cfg;
        ref.order = false;

        // A THROWAWAY CALL FIRST, because the first one in the process is not
        // measuring the traversal. It pays for the shader compile, the buffer
        // allocations and the clocks coming up, and on Sponza that is 41% --
        // four times any effect being looked for here. Ratios taken against a
        // cold first measurement credited the warm-up to whichever variant
        // happened to run second.
        // INTERLEAVED REPEATS, AND THE MINIMUM OF EACH -- not a single A/B.
        //
        // Two things make a single pair of calls unable to measure a 1.3x
        // effect here. The first call in a process is ~40% slow: shader
        // compile, allocations, and the clocks coming up. And absolute
        // throughput varies by up to 74% BETWEEN runs of the same binary --
        // one run measured this scene's baseline at 5109 ms and another, while
        // the machine was throttled, at 8728, which compressed a real 1.33x
        // down to an apparent 1.08x.
        //
        // Neither is fixed by a first-versus-last drift check, which reported
        // +0.2% on the throttled run. Interleaving the configurations and
        // keeping each one's MINIMUM is what works: the minimum is the rep
        // where least else interfered, and interleaving means any drift over
        // the run is charged to both configurations rather than to whichever
        // one happened to go second. Every sample is printed so a suspicious
        // pattern stays visible instead of being averaged into the answer.
        //
        // A warning against over-reading those samples, from this gate's own
        // history: a run of five that went slow/fast/slow/fast/slow looked
        // exactly like position-dependent noise, and was mistaken for it. The
        // fast ones were the two configurations with the ordering ON. Lining
        // samples up by configuration before by position is the discipline.
        const int kReps = 3;
        double warm_ms = 0.0;
        (void)timed(ref, warm_ms);

        SolveConfig on = cfg;
        on.order = true;

        double best_off = 1e30, best_on = 1e30;
        std::vector<uint32_t> want, got;
        printf("[GATE] 9 order   %u cameras x %u texels on %zu triangles, "
               "%u clusters in %u groups (warm-up %.0f ms discarded)\n",
               kCams, texels_o, tris.size(), scene.cluster_count(),
               scene.group_count(), warm_ms);
        for (int rep = 0; rep < kReps; ++rep) {
            double a = 0.0, b = 0.0;
            std::vector<uint32_t> va = timed(ref, a);
            std::vector<uint32_t> vb = timed(on, b);
            best_off = std::min(best_off, a);
            best_on = std::min(best_on, b);
            printf("[GATE] 9 order   rep %d       index order %8.1f ms   "
                   "nearest first %8.1f ms\n", rep, a, b);
            if (rep == 0) { want = std::move(va); got = std::move(vb); }
        }

        {
            // A DIFFERENT TRIANGLE IS NOT YET A DIFFERENT ANSWER.
            //
            // `dist < bd` is strict, so among triangles at exactly the same
            // distance the one found FIRST wins -- and changing the visit order
            // is precisely what this gate varies. Sponza carries coincident and
            // duplicated polygons in quantity, so a tie flipping is the
            // expected, harmless outcome and says nothing about the bound.
            //
            // What would be a bug is a bound that pruned a box which could have
            // won: that shows up as the two winners lying at MEASURABLY
            // different distances. So intersect both and compare the depths --
            // only a real difference counts against the gate.
            std::size_t diff = 0, ties = 0;
            double worst_rel = 0.0;
            const std::size_t n = std::min(want.size(), got.size());
            for (std::size_t i = 0; i < n; ++i) {
                if (want[i] == got[i]) continue;
                const uint32_t c = uint32_t(i / texels_o), tx = uint32_t(i % texels_o);
                glm::vec3 N = glm::normalize(glm::vec3(nrm[c])), T, B;
                onb(N, T, B);
                const glm::vec3 P = glm::vec3(pos[c]) + N * cfg.bias;
                const glm::vec3 dl = glm::vec3(qo.texels[tx]);
                const glm::vec3 d = T * dl.x + B * dl.y + N * dl.z;
                float ta = 0.0f, tb = 0.0f;
                const bool ha = want[i] != 0xFFFFFFFFu && ray_tri(P, d, tris[want[i]], ta);
                const bool hb = got[i]  != 0xFFFFFFFFu && ray_tri(P, d, tris[got[i]],  tb);
                if (!ha || !hb) { ++diff; continue; }
                const double rel = std::abs(double(ta) - double(tb)) /
                                   std::max(1e-9, double(std::max(ta, tb)));
                if (rel <= 1e-5) { ++ties; continue; }
                worst_rel = std::max(worst_rel, rel);
                ++diff;
            }
            if (want.size() != got.size()) diff = std::max(want.size(), got.size());
            r.check(9, "order", "nearest first", double(diff), 0.0, 0.0, true);
            printf("[GATE] 9 order   %zu tie(s) at equal depth, %zu at different "
                   "depth (worst rel %.3g)\n", ties, diff, worst_rel);
            printf("[GATE] 9 order   best of %d: %8.1f -> %8.1f ms   %.2fx\n",
                   kReps, best_off, best_on, best_off / std::max(1e-9, best_on));
        }
    }

    printf("[GATE] %d passed, %d failed\n", r.passed, r.failed);
    return r.failed == 0;
}

} // namespace mbg

#include "validate.hpp"

#include <gllib/log.hpp>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace sgi {
namespace {

constexpr double kPi = 3.14159265358979323846;

int g_pass = 0, g_fail = 0;

void report(const char* name, double got, double expect, double tol) {
    const double rel = expect != 0.0 ? std::abs(got - expect) / std::abs(expect)
                                     : std::abs(got - expect);
    const bool ok = rel <= tol;
    ok ? ++g_pass : ++g_fail;
    std::printf("[GATE] %-34s got %-14.8g expect %-14.8g rel %-10.3g tol %-8.3g %s\n",
                name, got, expect, rel, tol, ok ? "PASS" : "FAIL");
    std::fflush(stdout);
}

void note(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    std::printf("[GATE]   ");
    std::vprintf(fmt, ap);
    std::printf("\n");
    va_end(ap);
    std::fflush(stdout);
}

// A configuration with every softening and bias that would bias an analytic
// comparison turned off.
SolveConfig exact(const SolveConfig& base) {
    SolveConfig c = base;
    c.soft_eps = 0.0f;
    c.plane_bias = 1.0f;
    c.horizon = 0.0f;
    c.emissive_scale = 1.0f;
    c.sky = glm::vec3(0.0f);
    c.two_sided = false;
    c.rotate = 0;                 // any per-surfel rotation makes runs incomparable
    c.no_occlusion = false;
    c.debug_constant = false;
    c.max_sweeps = 0;
    c.running = true;
    return c;
}

double mean_luma(const std::vector<glm::vec4>& v) {
    double s = 0.0;
    for (const glm::vec4& e : v) s += 0.2126 * e.r + 0.7152 * e.g + 0.0722 * e.b;
    return v.empty() ? 0.0 : s / double(v.size());
}

// --- gate 1: point-to-point -------------------------------------------------

void gate_p2p(SurfelSet& set, Solver& solver, const SolveConfig& base) {
    // Emitter at the origin facing +z; receiver placed so cos_E = 0.8 and
    // cos_P = 1.0. Deliberately UNEQUAL, so swapping the two cosines is
    // detectable -- with equal cosines the most common normalization bug in
    // this method passes silently.
    const float r = 0.01f;
    const std::vector<glm::vec4> pr = {
        glm::vec4(0.0f, 0.0f, 0.0f, r),
        glm::vec4(0.3f, 0.0f, 0.4f, r),
    };
    const std::vector<glm::vec3> nr = {
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(-0.6f, 0.0f, -0.8f),
    };
    const std::vector<glm::vec3> alb = {glm::vec3(0.0f), glm::vec3(0.0f)};
    const std::vector<glm::vec3> emi = {glm::vec3(1.0f), glm::vec3(0.0f)};

    const double a = kPi * double(r) * double(r);
    const double expect = a * 1.0 * 0.8 * 1.0 / 0.25;   // a * L * cosE * cosP / d^2

    SolveConfig c = exact(base);
    for (int m = 0; m < 2; ++m) {
        c.method = m == 0 ? Method::Radiance : Method::Micro;
        set.build_explicit(pr, nr, alb, emi);
        solver.reset(set);
        solver.run_sweeps(set, c, 1, 0);
        const std::vector<glm::vec4> E = set.read_irradiance();
        report(m == 0 ? "1 p2p  E via M1 (radiance)" : "1 p2p  E via M2 (microbuffer)",
               double(E[1].r), expect, m == 0 ? 1e-5 : 0.02);
        report(m == 0 ? "1 p2p  emitter receives 0 (M1)" : "1 p2p  emitter receives 0 (M2)",
               double(E[0].r), 0.0, 1e-9);
    }
}

// --- gate 2: coaxial parallel discs -----------------------------------------

// Tiles a disc of radius R with `m` surfels on a Vogel (sunflower) spiral, which
// is uniform in density and fully deterministic.
void tile_disc(std::vector<glm::vec4>& pr, std::vector<glm::vec3>& nr,
               std::vector<glm::vec3>& alb, std::vector<glm::vec3>& emi,
               float R, float z, const glm::vec3& n, uint32_t m,
               const glm::vec3& albedo, const glm::vec3& emission) {
    // Area-preserving: m * pi * rs^2 == pi * R^2.
    const float rs = R / std::sqrt(float(m));
    const float ga = 2.39996322972865332f;   // golden angle
    for (uint32_t k = 0; k < m; ++k) {
        const float rad = R * std::sqrt((float(k) + 0.5f) / float(m));
        const float th  = float(k) * ga;
        pr.push_back(glm::vec4(rad * std::cos(th), rad * std::sin(th), z, rs));
        nr.push_back(n);
        alb.push_back(albedo);
        emi.push_back(emission);
    }
}

// The BAKE'S sampler, over a quad, so gate 12 can measure what the real scene
// has rather than what a nicer point set would give. Mirrors SurfelSet::build:
// R2 per triangle with a Cranley-Patterson rotation, one global radius chosen so
// N*pi*r^2 == A.
//
// This matters more than it sounds. A Vogel spiral is far more locally uniform
// than R2-folded-into-a-triangle, and the two need DIFFERENT occluder inflation
// to seal the same nominal coverage -- so a gate built on a spiral will pass
// while the render still leaks. That is exactly what happened.
void tile_quad_r2(std::vector<glm::vec4>& pr, std::vector<glm::vec3>& nr,
                  std::vector<glm::vec3>& alb, std::vector<glm::vec3>& emi,
                  float R, float z, const glm::vec3& n, uint32_t m,
                  const glm::vec3& albedo, const glm::vec3& emission) {
    const double a1 = 0.7548776662466927, a2 = 0.5698402909980532;
    const float area = 4.0f * R * R;
    const float rs = std::sqrt(area / (float(kPi) * float(m)));
    // Two triangles of a 2R x 2R square, each getting half the samples.
    const glm::vec3 c0(-R, -R, z), c1(R, -R, z), c2(R, R, z), c3(-R, R, z);
    const glm::vec3 tri[2][3] = {{c0, c1, c2}, {c0, c2, c3}};
    for (int t = 0; t < 2; ++t) {
        const glm::vec3 e1 = tri[t][1] - tri[t][0], e2 = tri[t][2] - tri[t][0];
        const double ox = t == 0 ? 0.317 : 0.719, oy = t == 0 ? 0.845 : 0.203;
        for (uint32_t k = 0; k < m / 2u; ++k) {
            auto fr = [](double x) { return float(x - std::floor(x)); };
            float u = fr(ox + a1 * double(k + 1));
            float v = fr(oy + a2 * double(k + 1));
            if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
            pr.push_back(glm::vec4(tri[t][0] + e1 * u + e2 * v, rs));
            nr.push_back(n);
            alb.push_back(albedo);
            emi.push_back(emission);
        }
    }
}

// The same R2 sampler over an arbitrary parallelogram, so a gate can build a
// plane in any orientation. tile_quad_r2 is the axis-aligned special case and
// stays as it is, because gate 12's numbers were measured with those exact
// Cranley-Patterson offsets.
void tile_rect_r2(std::vector<glm::vec4>& pr, std::vector<glm::vec3>& nr,
                  std::vector<glm::vec3>& alb, std::vector<glm::vec3>& emi,
                  const glm::vec3& o, const glm::vec3& e1, const glm::vec3& e2,
                  const glm::vec3& n, uint32_t m,
                  const glm::vec3& albedo, const glm::vec3& emission) {
    const double a1 = 0.7548776662466927, a2 = 0.5698402909980532;
    const float area = glm::length(glm::cross(e1, e2));
    const float rs = std::sqrt(area / (float(kPi) * float(m)));
    // Two triangles of the parallelogram, each with its own rotation, exactly as
    // the bake sees a quad that arrived as two mesh triangles.
    const glm::vec3 tri[2][3] = {{o, o + e1, o + e1 + e2}, {o, o + e1 + e2, o + e2}};
    for (int t = 0; t < 2; ++t) {
        const glm::vec3 d1 = tri[t][1] - tri[t][0], d2 = tri[t][2] - tri[t][0];
        const double ox = t == 0 ? 0.317 : 0.719, oy = t == 0 ? 0.845 : 0.203;
        for (uint32_t k = 0; k < m / 2u; ++k) {
            auto fr = [](double x) { return float(x - std::floor(x)); };
            float u = fr(ox + a1 * double(k + 1));
            float v = fr(oy + a2 * double(k + 1));
            if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
            pr.push_back(glm::vec4(tri[t][0] + d1 * u + d2 * v, rs));
            nr.push_back(n);
            alb.push_back(albedo);
            emi.push_back(emission);
        }
    }
}

void gate_disc(SurfelSet& set, Solver& solver, const SolveConfig& base) {
    const float R = 0.5f;
    const float L = 1.0f;
    const uint32_t m = 4096;
    const double xs[3] = {0.5, 1.0, 2.0};

    for (int mi = 0; mi < 2; ++mi) {
        SolveConfig c = exact(base);
        c.method = mi == 0 ? Method::Radiance : Method::Micro;
        for (double x : xs) {
            const float h = float(x) * R;
            std::vector<glm::vec4> pr; std::vector<glm::vec3> nr, alb, emi;
            tile_disc(pr, nr, alb, emi, R, 0.0f, glm::vec3(0, 0, 1), m,
                      glm::vec3(0.0f), glm::vec3(L));                  // emitter
            tile_disc(pr, nr, alb, emi, R, h, glm::vec3(0, 0, -1), m,
                      glm::vec3(0.0f), glm::vec3(0.0f));               // receiver

            set.build_explicit(pr, nr, alb, emi);
            solver.reset(set);
            solver.run_sweeps(set, c, 1, 0);
            const std::vector<glm::vec4> E = set.read_irradiance();

            // Equal coaxial discs: F = 1 + x^2/2 - (x/2)sqrt(x^2+4).
            // Average irradiance on disc 2 is pi*L*F, since B = pi*L and the
            // areas are equal.
            const double F = 1.0 + x * x * 0.5 - (x * 0.5) * std::sqrt(x * x + 4.0);
            double sum = 0.0;
            for (uint32_t k = m; k < 2 * m; ++k) sum += double(E[k].r);
            const double got = sum / double(m);

            char name[96];
            std::snprintf(name, sizeof name, "2 disc %s  h/R = %.1f",
                          mi == 0 ? "M1" : "M2", x);
            report(name, got, kPi * double(L) * F, mi == 0 ? 0.01 : 0.05);
        }
    }
}

// --- gate 3: constant-radiance hemisphere -----------------------------------

void gate_hemi(SurfelSet& set, Solver& solver, const SolveConfig& base) {
    double sdw = 0.0, swcos = 0.0;
    for (uint32_t ms : {8u, 16u}) {
        solver.bucket_sums(ms, sdw, swcos);
        char n1[64], n2[64];
        std::snprintf(n1, sizeof n1, "3 bucket %ux%u  sum dw", ms, ms);
        std::snprintf(n2, sizeof n2, "3 bucket %ux%u  sum wcos", ms, ms);
        // These fail DIFFERENTLY: sum dw is a solid-angle bug, sum wcos with
        // sum dw passing is a direction bug.
        report(n1, sdw, 2.0 * kPi, 1e-5);
        report(n2, swcos, kPi, 1e-5);
    }

    // Force every bucket to a constant radiance, bypassing coverage and the
    // depth resolve entirely, so E must be exactly the cosine integral: pi * L.
    const std::vector<glm::vec4> pr = {glm::vec4(0, 0, 0, 0.01f), glm::vec4(1, 0, 0, 0.01f)};
    const std::vector<glm::vec3> nr = {glm::vec3(0, 0, 1), glm::vec3(0, 0, 1)};
    const std::vector<glm::vec3> z2 = {glm::vec3(0.0f), glm::vec3(0.0f)};
    set.build_explicit(pr, nr, z2, z2);

    double got[2] = {0.0, 0.0};
    const float Ls[2] = {0.5f, 2.0f};
    for (int i = 0; i < 2; ++i) {
        SolveConfig c = exact(base);
        c.method = Method::Micro;
        c.debug_constant = true;
        c.debug_radiance = Ls[i];
        solver.reset(set);
        solver.run_sweeps(set, c, 1, 0);
        got[i] = double(set.read_irradiance()[0].r);
        char name[64];
        std::snprintf(name, sizeof name, "3 hemi  E for constant L = %.1f", double(Ls[i]));
        report(name, got[i], kPi * double(Ls[i]), 2e-3);
    }
    report("3 hemi  linearity (L=2 / L=0.5)", got[0] != 0.0 ? got[1] / got[0] : 0.0,
           4.0, 1e-6);
}

// --- gate 4: energy scaling linearity ---------------------------------------

void gate_scale(SurfelSet& scene, Solver& solver, const SolveConfig& base,
                const std::vector<Tri>& tris, uint32_t surfels) {
    scene.build(tris, surfels);
    for (uint32_t sweeps : {1u, 8u}) {
        double m1 = 0.0, m8 = 0.0;
        SolveConfig c = base;
        c.rotate = 0;
        c.max_sweeps = 0;
        c.running = true;

        c.emissive_scale = 1.0f;
        solver.reset(scene);
        solver.run_sweeps(scene, c, sweeps, 0);
        m1 = mean_luma(scene.read_irradiance());

        c.emissive_scale = 8.0f;
        solver.reset(scene);
        solver.run_sweeps(scene, c, sweeps, 0);
        m8 = mean_luma(scene.read_irradiance());

        char name[80];
        std::snprintf(name, sizeof name, "4 scale  8x emission, %u sweep(s)", sweeps);
        // 0.1% -- the operator is exactly linear in the source, so the only
        // things that can break this are a clamp or a fixed-point overflow, and
        // the x8 case is specifically what pushes bf_micro's Q16.16 accumulator.
        report(name, m1 > 0.0 ? m8 / m1 : 0.0, 8.0, 1e-3);
    }
}

// --- gate 5: M1 vs M2 with occlusion disabled -------------------------------

void gate_noocc(SurfelSet& scene, Solver& solver, const SolveConfig& base,
                const std::vector<Tri>& tris, uint32_t surfels) {
    scene.build(tris, surfels);

    SolveConfig c = base;
    c.rotate = 0;
    c.max_sweeps = 0;
    c.running = true;
    c.sky = glm::vec3(0.0f);
    c.no_occlusion = false;

    c.method = Method::Radiance;
    solver.reset(scene);
    solver.run_sweeps(scene, c, 1, 0);
    const std::vector<glm::vec4> a = scene.read_irradiance();

    c.method = Method::Micro;
    c.no_occlusion = true;      // accept every candidate, no coverage clamp
    solver.reset(scene);
    solver.run_sweeps(scene, c, 1, 0);
    const std::vector<glm::vec4> b = scene.read_irradiance();

    double sa = 0.0, sd = 0.0, worst = 0.0;
    std::size_t worst_i = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double la = 0.2126 * a[i].r + 0.7152 * a[i].g + 0.0722 * a[i].b;
        const double lb = 0.2126 * b[i].r + 0.7152 * b[i].g + 0.0722 * b[i].b;
        sa += la;
        sd += std::abs(la - lb);
        const double rel = la > 1e-6 ? std::abs(la - lb) / la : 0.0;
        if (rel > worst) { worst = rel; worst_i = i; }
    }
    note("worst surfel %zu: M1 %.6g vs M2 %.6g (rel %.3g)", worst_i,
         double(a[worst_i].r), double(b[worst_i].r), worst);
    char name[80];
    std::snprintf(name, sizeof name, "5 noocc  M1 vs M2 at %ux%u buckets", base.ms, base.ms);
    // This is the only gate that tests ANGULAR PLACEMENT rather than
    // its total energy: a wrong footprint radius, a transposed tangent frame, or
    // a wrong per-bucket dw all show up here and nowhere else.
    report(name, sa > 0.0 ? sd / sa : 1.0, 0.0, base.ms >= 16 ? 0.02 : 0.05);
}

// --- gate 8: occlusion against a CPU ray cast -------------------------------
//
// The gates above are all analytic and all UNOCCLUDED: p2p and disc have two
// surfaces and nothing between them, hemi has no geometry at all, and noocc
// forces every candidate visible on purpose. Nothing in the set actually
// measured whether the microbuffer OCCLUDES correctly -- which is the entire
// content of section 10 step 2.
//
// So: light the scene with a uniform unit sky and no emitters. Then a surfel's
// irradiance is exactly
//
//     E = INTEGRAL over the unoccluded hemisphere of cos(theta) dOmega
//       = PI * (cosine-weighted fraction of the hemisphere that is open)
//
// and the same quantity can be ray cast against the 32 source triangles on the
// CPU to any precision wanted. The comparison needs no reference image, no
// tone curve and no camera.
//
// Reported per openness band, because a uniform bias and a bias concentrated in
// the occluded regions are different bugs and the mean hides the difference.

// Moller-Trumbore. Back faces hit too: an occluder occludes from both sides.
bool ray_tri(const glm::vec3& o, const glm::vec3& d, const Tri& t, float tmax) {
    const glm::vec3 e1 = t.p[1] - t.p[0], e2 = t.p[2] - t.p[0];
    const glm::vec3 pv = glm::cross(d, e2);
    const float det = glm::dot(e1, pv);
    if (std::abs(det) < 1e-12f) return false;
    const float inv = 1.0f / det;
    const glm::vec3 tv = o - t.p[0];
    const float u = glm::dot(tv, pv) * inv;
    if (u < 0.0f || u > 1.0f) return false;
    const glm::vec3 qv = glm::cross(tv, e1);
    const float v = glm::dot(d, qv) * inv;
    if (v < 0.0f || u + v > 1.0f) return false;
    const float tt = glm::dot(e2, qv) * inv;
    return tt > 1e-4f && tt < tmax;
}

// Cosine-weighted openness at (p, n): the fraction of INTEGRAL(cos dOmega) that
// reaches infinity. Cosine-distributed directions, so the estimator is the plain
// hit fraction and no weights are needed.
double cpu_openness(const glm::vec3& p, const glm::vec3& n,
                    const std::vector<Tri>& tris, uint32_t rays, uint32_t seed) {
    // Duff ONB, matching bf_micro.comp exactly.
    const float sg = n.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -1.0f / (sg + n.z);
    const float c = n.x * n.y * a;
    const glm::vec3 T(1.0f + sg * n.x * n.x * a, sg * c, -sg * n.x);
    const glm::vec3 B(c, sg + n.y * n.y * a, -n.y);

    const glm::vec3 o = p + n * 1e-3f;
    uint32_t h = seed * 2654435761u + 1u;
    auto rnd = [&h]() {
        h ^= h << 13; h ^= h >> 17; h ^= h << 5;
        return float(h) * (1.0f / 4294967296.0f);
    };

    uint32_t open = 0;
    for (uint32_t i = 0; i < rays; ++i) {
        // Stratified in the first dimension, jittered in the second.
        const float u1 = (float(i) + rnd()) / float(rays);
        const float u2 = rnd();
        const float sr = std::sqrt(u1), ph = 6.28318530718f * u2;
        const glm::vec3 d = T * (sr * std::cos(ph)) + B * (sr * std::sin(ph)) +
                            n * std::sqrt(std::max(0.0f, 1.0f - u1));
        bool hit = false;
        for (const Tri& t : tris) { if (ray_tri(o, d, t, 1e30f)) { hit = true; break; } }
        if (!hit) ++open;
    }
    return double(open) / double(rays);
}

void gate_occ(SurfelSet& scene, Solver& solver, const SolveConfig& base,
              const std::vector<Tri>& tris, uint32_t surfels) {
    scene.build(tris, surfels);

    SolveConfig c = base;
    c.method = Method::Micro;
    c.sky = glm::vec3(1.0f);        // unit sky ...
    c.emissive_scale = 0.0f;        // ... and nothing else emitting
    c.rotate = 0;
    c.no_occlusion = false;
    c.debug_constant = false;
    c.max_sweeps = 0;
    c.running = true;

    solver.reset(scene);
    solver.run_sweeps(scene, c, 1, 0);
    const std::vector<glm::vec4> E = scene.read_irradiance();

    const std::vector<glm::vec4>& pr = scene.pos_rad();
    const std::vector<uint32_t>&  nn = scene.normal();

    // A deterministic stride over the whole set rather than a prefix: the bake
    // emits surfels triangle by triangle, so a prefix would be one wall.
    const uint32_t kSamples = 4000, kRays = 2048;
    const uint32_t stride = std::max(1u, scene.count() / kSamples);

    struct Band { const char* name; double lo, hi; double sum_gpu, sum_cpu, sum_abs; int n; };
    Band bands[] = {
        {"deeply occluded  (<0.15)", 0.00, 0.15, 0, 0, 0, 0},
        {"occluded    (0.15-0.35)",  0.15, 0.35, 0, 0, 0, 0},
        {"half open   (0.35-0.60)",  0.35, 0.60, 0, 0, 0, 0},
        {"mostly open (0.60-1.00)",  0.60, 1.01, 0, 0, 0, 0},
    };
    double sum_abs = 0.0, sum_gpu = 0.0, sum_cpu = 0.0;
    double worst = 0.0; glm::vec3 worst_p(0.0f); double worst_g = 0, worst_c = 0;
    int n = 0;

    for (uint32_t i = 0; i < scene.count(); i += stride) {
        const glm::vec3 p(pr[i]);
        const glm::vec3 nv = unpack_oct_snorm16(nn[i]);
        const double gpu = double(E[i].r) / kPi;                 // sky is white
        const double cpu = cpu_openness(p, nv, tris, kRays, i + 1u);
        const double e = std::abs(gpu - cpu);
        sum_abs += e; sum_gpu += gpu; sum_cpu += cpu; ++n;
        if (e > worst) { worst = e; worst_p = p; worst_g = gpu; worst_c = cpu; }
        for (Band& b : bands)
            if (cpu >= b.lo && cpu < b.hi) {
                b.sum_gpu += gpu; b.sum_cpu += cpu; b.sum_abs += e; ++b.n; break;
            }
    }
    if (n == 0) return;

    note("openness over %d of %u surfels at %ux%u buckets, %u rays each "
         "(cosine-weighted, 0 = enclosed)", n, scene.count(), c.ms, c.ms, kRays);
    note("%-26s %8s %8s %8s %7s", "band (by CPU openness)", "CPU", "GPU", "mean|d|", "count");
    for (const Band& b : bands) {
        if (b.n == 0) continue;
        note("%-26s %8.4f %8.4f %8.4f %7d", b.name, b.sum_cpu / b.n, b.sum_gpu / b.n,
             b.sum_abs / b.n, b.n);
    }
    note("worst surfel at (%.3f, %.3f, %.3f): GPU %.4f vs CPU %.4f",
         double(worst_p.x), double(worst_p.y), double(worst_p.z), worst_g, worst_c);

    // Asserted against the SURFEL SAMPLING FLOOR rather than a fixed number.
    //
    // What is left once the transport is right is a point-sampling artifact: a
    // texel's coverage is a sum over the surfel centres that land in it, so it
    // fluctuates about 1 on a fully covered texel, and clamping at 1 removes the
    // excess while keeping the deficit. That bias goes as 1/sqrt(surfels per
    // texel), hence 1/sqrt(N). Measured on this scene at 16x16:
    //
    //     N = 15000   0.0569        N = 30000   0.0420        N = 60000   0.0316
    //
    // i.e. 0.74x per doubling against the 0.707x the model predicts, and
    // err*sqrt(N) = 6.97 / 7.27 / 7.74 -- flat, which is the point. A tolerance
    // of 12/sqrt(N) sits ~1.6x above that floor, so it passes at any density
    // while still failing hard on anything structural: the same-surface filter
    // bug this gate was written to find sat at 25/sqrt(N) and did not improve
    // with N at all, which is exactly how it was identified.
    const double floor_tol = 12.0 / std::sqrt(double(scene.count()));
    note("surfel sampling floor at N = %u: tolerance %.4f (12/sqrt(N))",
         scene.count(), floor_tol);
    report("8 occ  mean |openness error|", sum_abs / n, 0.0, floor_tol);
    for (const Band& b : bands) {
        if (b.n == 0 || b.hi > 0.16) continue;
        report("8 occ  deeply-occluded bias", b.sum_gpu / b.n - b.sum_cpu / b.n, 0.0,
               16.0 / std::sqrt(double(scene.count())));
    }
    note("population mean: CPU %.4f, GPU %.4f (%+.1f%%)", sum_cpu / n, sum_gpu / n,
         (sum_gpu / sum_cpu - 1.0) * 100.0);
}

// --- gate 9: an opaque blocker is opaque ------------------------------------
//
// Three parallel discs on one axis: an emitter, a larger opaque blocker fully
// shadowing it, and a receiver behind the blocker facing its BACK.
//
//     emitter (L=1) ---> | blocker (albedo 0.8) | ---> receiver
//        z = 0                  z = h                   z = 2h
//
// The receiver must get essentially nothing. It needs at least two sweeps to
// mean anything: on sweep 1 the blocker's outgoing radiance is still zero, so
// nothing can leak through it and the test passes vacuously. Only once the
// blocker is LIT does the question "does a lit surface re-emit out of its back?"
// have an answer.
//
// This gate exists because none of gates 1-8 could see the bug it was written
// for. They are all either unoccluded by construction or measure occlusion with
// a uniform sky (gate 8), which is a pure visibility question that says nothing
// about which face of a surfel emits. The bug -- `cosE = abs(cosE)` applied to
// every surfel -- put a lit surface's radiance out of its own back, lighting the
// sealed volume under each Cornell box through its lid. In the image it read as
// a one-pixel bright line exactly along the box/floor contact, which whole-image
// means and MAE are entirely insensitive to.

void gate_opaque(SurfelSet& set, Solver& solver, const SolveConfig& base) {
    const uint32_t m = 3000;
    const float R_small = 0.3f, R_block = 1.0f, h = 0.5f;

    std::vector<glm::vec4> pr; std::vector<glm::vec3> nr, alb, emi;
    tile_disc(pr, nr, alb, emi, R_small, 0.0f,      glm::vec3(0, 0,  1), m,
              glm::vec3(0.0f), glm::vec3(1.0f));                    // [0, m)   emitter
    tile_disc(pr, nr, alb, emi, R_block, h,         glm::vec3(0, 0, -1), m,
              glm::vec3(0.8f), glm::vec3(0.0f));                    // [m, 2m)  blocker
    tile_disc(pr, nr, alb, emi, R_small, 2.0f * h,  glm::vec3(0, 0, -1), m,
              glm::vec3(0.8f), glm::vec3(0.0f));                    // [2m, 3m) receiver

    SolveConfig c = exact(base);
    c.method = Method::Micro;
    // Deliberately NOT forced: this gate exists to catch a wrong two-sidedness
    // setting, so it has to see the live one. exact() zeroes it like every other
    // bias, which would make the gate unable to fail -- verified by running it
    // with SGI_TWOSIDED=1, which must report a large leak.
    c.two_sided = base.two_sided;
    set.build_explicit(pr, nr, alb, emi);
    solver.reset(set);
    solver.run_sweeps(set, c, 3, 0);
    const std::vector<glm::vec4> E = set.read_irradiance();

    double blocked = 0.0, behind = 0.0;
    for (uint32_t k = m;        k < 2 * m; ++k) blocked += double(E[k].r);
    for (uint32_t k = 2u * m;   k < 3 * m; ++k) behind  += double(E[k].r);
    blocked /= double(m); behind /= double(m);

    note("blocker lit to E = %.5f; receiver behind it gets E = %.5f (%.2f%%)",
         blocked, behind, blocked > 0.0 ? 100.0 * behind / blocked : 0.0);
    report("9 opaque  blocker is lit at all", blocked > 1e-4 ? 1.0 : 0.0, 1.0, 1e-9);
    report("9 opaque  leak through the blocker",
           blocked > 0.0 ? behind / blocked : 0.0, 0.0, 0.02);
}

// --- gate 12: grazing incidence, the case Cornell cannot present -------------
//
// EVERY NEE PARAMETER IS DIMENSIONLESS -- thickness, bias and the same-surface
// tolerance are all in SURFEL RADII, and self_cos is a cosine -- so none of them
// carries a scene scale and none can be "tuned to Cornell" in the units sense.
// What they CAN be tuned to is Cornell's geometry, and Cornell presents exactly
// one lighting configuration: a small panel directly overhead, every surface
// flat, every dihedral 90 degrees, and no ray between a receiver and the light
// ever crosses a surface at a shallow angle.
//
// The disc-soup occluder model is at its weakest precisely there. A surfel seen
// edge-on projects to a line and occludes nothing, so a single-layer wall goes
// transparent as the line of sight flattens into its plane. `thick` is what
// stops that -- and Cornell never tests it, because its walls are only ever
// crossed head-on.
//
// So: an opaque single-layer wall, a receiver on one side and an emitter on the
// other, and the whole configuration rotated from head-on (theta = 0) to nearly
// in the wall's plane (theta = 85). The leak at each angle is the fraction of
// the unoccluded irradiance that gets through. It must stay near zero at every
// angle, for one value of `thick`, or the parameter is fitted to the geometry
// rather than to the model.
//
// The second sweep is COVERAGE. The estimator assumes the bake tiles the surface
// at sum(pi r^2)/A == 1, which gate 6 asserts and which the baker guarantees --
// until the set hits the 65535-surfel ceiling and gets TRUNCATED, at which point
// the surface has actual holes in it. CornellBoxBunnyMirror.glb needs 98829
// surfels and lands at coverage 0.663; this sweep is what says how much light
// that costs, and it is not something `thick` can compensate for.

void gate_graze(SurfelSet& set, Solver& solver, const SolveConfig& base) {
    const float R_w = 0.6f;      // wall radius
    const float t   = 1.0f;      // receiver and emitter distance from the wall centre
    const float hu = 0.15f, hv = 0.15f;
    const float L  = 3.0f;
    const uint32_t m_wall = 6000;

    const double thetas[] = {0.0, 45.0, 70.0, 80.0, 85.0};
    // (sampler, coverage, occluder scale, thickness). Not a full cross product:
    // each row exists to isolate one thing.
    struct Row { const char* sampler; double cov, occ, thick; };
    const Row rows[] = {
        {"spiral", 1.00, 1.00, 0.25},   // the uniform point set, under-inflated
        {"spiral", 1.00, 1.25, 0.25},   // ... seals here, which misled me
        {"r2",     1.00, 1.00, 0.25},   // the BAKE'S sampler: same coverage, worse
        {"r2",     1.00, 1.25, 0.25},   // ... still leaking at the spiral's answer
        {"r2",     1.00, 1.50, 0.25},
        {"r2",     1.00, 2.00, 0.25},   // shipped
        {"r2",     0.66, 2.00, 0.25},   // a bake truncated at the 65535 ceiling
        {"r2",     1.00, 2.00, 0.00},   // a flat disc is measure zero: blocks nothing
    };

    std::printf("[GATE]   grazing leak: %% of the unoccluded direct term that gets "
                "through a single-layer wall\n");
    std::printf("[GATE]   %-30s", "sampler / config");
    for (double th : thetas) std::printf("%8.0f", th);
    std::printf("\n");

    double worst_ref = 0.0;
    for (const Row& row : rows) {
        {
        {
            const double cov = row.cov, occ = row.occ, thick = row.thick;
            const bool use_r2 = row.sampler[0] == 'r';
            char lbl[64];
            std::snprintf(lbl, sizeof lbl, "%-6s cov %.2f occ %.2f thk %.2f",
                          row.sampler, cov, occ, thick);
            std::printf("[GATE]   %-30s", lbl);

            for (double thd : thetas) {
                const float th = float(thd) * float(kPi) / 180.0f;
                const glm::vec3 dir(std::sin(th), 0.0f, std::cos(th));  // wall -> emitter
                const glm::vec3 ec =  dir * t;
                const glm::vec3 rp = -dir * t;

                // The emitter rectangle stays PARALLEL TO THE WALL rather than
                // square to the line of sight. Squared to the line of sight it
                // straddles the wall's plane at grazing angles -- half the light
                // ends up on the receiver's own side, where no wall could block
                // it -- and the gate then reports a 25% "leak" that is its own
                // geometry, not the estimator's. Keeping it parallel means every
                // ray from the receiver to the light crosses the wall exactly
                // once, at the angle being swept.
                const glm::vec3 eu(1, 0, 0), ev(0, 1, 0), en(0, 0, -1);
                std::vector<Tri> etris(2);
                const glm::vec3 c0 = ec - eu * hu - ev * hv, c1 = ec + eu * hu - ev * hv;
                const glm::vec3 c2 = ec + eu * hu + ev * hv, c3 = ec - eu * hu + ev * hv;
                etris[0].p[0] = c0; etris[0].p[1] = c1; etris[0].p[2] = c2;
                etris[1].p[0] = c0; etris[1].p[1] = c2; etris[1].p[2] = c3;
                for (Tri& tr : etris) {
                    tr.n = en;
                    tr.emission = glm::vec3(L);
                    tr.albedo = glm::vec3(0.0f);
                    tr.area = 0.5f * glm::length(glm::cross(tr.p[1] - tr.p[0],
                                                            tr.p[2] - tr.p[0]));
                }

                SolveConfig c = exact(base);
                c.method = Method::Micro;
                c.nee = true;
                c.nee_thick = float(thick);
                c.nee_occ   = float(occ);

                auto run = [&](bool with_wall) {
                    std::vector<glm::vec4> pr; std::vector<glm::vec3> nr, alb, emi;
                    if (with_wall) {
                        if (use_r2)
                            tile_quad_r2(pr, nr, alb, emi, R_w, 0.0f, glm::vec3(0, 0, 1),
                                         m_wall, glm::vec3(0.0f), glm::vec3(0.0f));
                        else
                            tile_disc(pr, nr, alb, emi, R_w, 0.0f, glm::vec3(0, 0, 1),
                                      m_wall, glm::vec3(0.0f), glm::vec3(0.0f));
                        // Coverage is the one quantity here that is NOT a free
                        // parameter: shrink the discs and the surface develops
                        // real holes, which no thickness model can close.
                        const float sc = float(std::sqrt(cov));
                        for (glm::vec4& p : pr) p.w *= sc;
                    }
                    const uint32_t side = 32;
                    const float cu = 2.0f * hu / float(side), cv = 2.0f * hv / float(side);
                    const float re = std::sqrt(cu * cv / float(kPi));
                    for (uint32_t j = 0; j < side; ++j)
                        for (uint32_t i = 0; i < side; ++i) {
                            const glm::vec3 p = ec + eu * (-hu + (float(i) + 0.5f) * cu)
                                                   + ev * (-hv + (float(j) + 0.5f) * cv);
                            pr.push_back(glm::vec4(p, re));
                            nr.push_back(en);
                            alb.push_back(glm::vec3(0.0f));
                            emi.push_back(glm::vec3(L));
                        }
                    const uint32_t rec = uint32_t(pr.size());
                    pr.push_back(glm::vec4(rp, 0.01f));
                    nr.push_back(dir);                 // facing the emitter
                    alb.push_back(glm::vec3(0.0f));
                    emi.push_back(glm::vec3(0.0f));

                    set.build_explicit(pr, nr, alb, emi);
                    SurfelGrid grid; grid.build(set, 1.0f);
                    EmitterSet em;   em.build(etris);
                    solver.attach(&grid, &em);
                    solver.reset(set);
                    solver.run_sweeps(set, c, 1, 0);
                    const double e = double(set.read_irradiance()[rec].r);
                    solver.attach(nullptr, nullptr);
                    return e;
                };

                const double open = run(false);
                const double shut = run(true);
                const double leak = open > 1e-9 ? 100.0 * shut / open : 0.0;
                std::printf("%8.2f", leak);
                // The assertion is made at the coverage the baker actually
                // produces and the thickness actually shipped.
                if (use_r2 && std::abs(cov - 1.0) < 1e-6 &&
                    std::abs(thick - 0.25) < 1e-6 && std::abs(occ - 2.0) < 1e-6)
                    worst_ref = std::max(worst_ref, leak);
            }
            std::printf("\n");
        }
        }
    }
    // The shipped configuration, at the coverage the baker actually produces,
    // must be opaque at EVERY angle -- not just head-on, which is the only one
    // the Cornell box ever presents.
    report("12 graze  worst leak, shipped config", worst_ref / 100.0, 0.0, 0.02);
}

// --- gate 13: the concave corner --------------------------------------------
//
// Cornell has twelve of these and the render has a defect on one of them: a
// horizontal band on the ceiling, one surfel row deep, at the ceiling/back-wall
// junction -- too dark on the junction row, too bright just above it, with three
// times the horizontal variation of the reference. Finding 32 established what
// it is NOT: not the reconstruction (invariant to every gather knob), not
// occlusion (M1 renders it identically), not the microbuffer, not density, not
// the bias or the horizon.
//
// That leaves the transport kernel itself at a concave crease, which Cornell
// cannot measure because it has no analytic answer anywhere. This gate builds
// the crease on its own: two unit squares meeting at a right angle along a
// shared edge, one a uniform emitter, the other a pure receiver, and NO other
// geometry. The receiver's irradiance then has a closed form -- Lambert's
// contour integral over the emitter rectangle, exact for a uniform-radiance
// polygon -- so every surfel can be checked against its own true value and
// binned by HEIGHT ABOVE THE CREASE IN RADII, which is the axis the defect
// lives on.
//
// Two sweeps, because they separate the two candidate causes:
//
//   * DENSITY. A quadrature artifact of the disc discretisation falls as the
//     surfels shrink; a structural cull or a wrongly-rejected neighbour does
//     not. Findings 30-32 already showed the band flat across 8k/30k/60k in the
//     real scene, so the prediction is "does not move" -- but that measurement
//     was against a reference image, and this one is against the exact answer.
//   * PLANE BIAS. `u_plane_bias` rejects any emitter whose centre lies within
//     one receiver radius of the receiver's tangent plane. That test exists to
//     stop a surfel lighting its own neighbours on a FLAT surface. At a concave
//     crease the perpendicular plane's near strip is also within that slab, and
//     it is the strip that carries most of the irradiance -- the 1/d^2 is
//     smallest there. If the band is the bias eating real transport, the error
//     is large at h ~ 1 radius, falls off as h grows, and moves with the bias.
void gate_corner(SurfelSet& set, Solver& solver, const SolveConfig& base) {
    const float S = 1.0f;        // square edge
    const float L = 3.0f;        // emitter radiance

    // The emitter rectangle, wound so that the contour integral comes out
    // positive for a receiver on the +x wall.
    const glm::dvec3 quad[4] = {glm::dvec3(0, 0, -0.5 * double(S)),
                                glm::dvec3(double(S), 0, -0.5 * double(S)),
                                glm::dvec3(double(S), 0, 0.5 * double(S)),
                                glm::dvec3(0, 0, 0.5 * double(S))};

    // E = L * (projected solid angle), and the projected solid angle of a planar
    // polygon is the Lambert contour integral
    //     Omega_p = 1/2 SUM_edges beta_i * (n . u_i)
    // with beta_i the angle the edge subtends at P and u_i the unit normal of
    // the triangle (P, v_i, v_i+1). Exact, not a quadrature: no discretisation
    // of the emitter enters the reference at all.
    //
    // No horizon clipping is needed here and that is by construction: the
    // emitter's near edge lies exactly IN the receiver's tangent plane, so the
    // polygon touches the horizon along one edge and never crosses it.
    auto exact_E = [&](const glm::dvec3& P, const glm::dvec3& n) {
        double s = 0.0;
        for (int i = 0; i < 4; ++i) {
            const glm::dvec3 a = quad[i] - P, b = quad[(i + 1) % 4] - P;
            const glm::dvec3 c = glm::cross(a, b);
            const double lc = glm::length(c);
            if (lc < 1e-300) continue;
            const double ct = glm::clamp(glm::dot(glm::normalize(a), glm::normalize(b)),
                                         -1.0, 1.0);
            s += std::acos(ct) * glm::dot(n, c / lc);
        }
        return std::max(0.0, 0.5 * double(L) * s);
    };

    // Bin edges in receiver radii. The defect is one surfel row deep, so the
    // first bins have to be one radius wide.
    const double edge[] = {0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 1e30};
    const int nb = int(sizeof edge / sizeof edge[0]) - 1;

    std::printf("[GATE]   concave corner: two unit squares at 90 deg, one emitting,\n");
    std::printf("[GATE]   receiver irradiance vs the exact polygon integral, by height\n");
    std::printf("[GATE]   above the crease in surfel radii. err%% = mean(got)/mean(exact)-1.\n");

    double worst_m1 = 0.0, worst_m2 = 0.0;   // worst |bias| beyond 4 radii

    // `exact()` zeroes the softening and the horizon so the kernel is measured
    // bare; the "shipped" rows put the defaults back, because those are the
    // numbers the render actually has.
    // `exact()` zeroes the softening and the horizon so the kernel is measured
    // bare; the "shipped" rows put the defaults back, because those are the
    // numbers the render actually has. Since the same-surface cull no longer
    // removes a receiver's coplanar neighbours on cosP alone, soft_eps = 0 is not
    // a configuration the solver has -- it diverges at contact by construction --
    // so every row here keeps the softening on.
    struct Run { const char* tag; uint32_t m; Method meth; float bias; float self_cos;
                 uint32_t ms; bool no_occ; };
    const Run runs[] = {
        {"M1  8k  shipped",   8192u, Method::Radiance, 1.0f, 0.90f, 16u, false},
        {"M1 32k  shipped",  32768u, Method::Radiance, 1.0f, 0.90f, 16u, false},
        {"M1  8k  bias 2.0",  8192u, Method::Radiance, 2.0f, 0.90f, 16u, false},
        {"M2  8k  shipped",   8192u, Method::Micro,    1.0f, 0.90f, 16u, false},
        {"M2 32k  shipped",  32768u, Method::Micro,    1.0f, 0.90f, 16u, false},
        {"M2  8k  8x8",       8192u, Method::Micro,    1.0f, 0.90f,  8u, false},
        {"M2  8k  no clamp",  8192u, Method::Micro,    1.0f, 0.90f, 16u, true},
    };

    std::printf("[GATE]   %-18s", "run \\ h in radii");
    for (int b = 0; b < nb; ++b) {
        char h[16];
        if (b + 1 == nb) std::snprintf(h, sizeof h, "%g+", edge[b]);
        else             std::snprintf(h, sizeof h, "%g-%g", edge[b], edge[b + 1]);
        std::printf("%9s", h);
    }
    std::printf("\n");

    for (const Run& run : runs) {
        std::vector<glm::vec4> pr; std::vector<glm::vec3> nr, alb, emi;
        // Emitter: the y = 0 floor, x in [0, S], facing +y.
        tile_rect_r2(pr, nr, alb, emi, glm::vec3(0, 0, -0.5f * S),
                     glm::vec3(S, 0, 0), glm::vec3(0, 0, S), glm::vec3(0, 1, 0),
                     run.m, glm::vec3(0.0f), glm::vec3(L));
        const uint32_t first_recv = uint32_t(pr.size());
        // Receiver: the x = 0 wall, y in [0, S], facing +x. Same crease.
        tile_rect_r2(pr, nr, alb, emi, glm::vec3(0, 0, -0.5f * S),
                     glm::vec3(0, S, 0), glm::vec3(0, 0, S), glm::vec3(1, 0, 0),
                     run.m, glm::vec3(0.0f), glm::vec3(0.0f));

        set.build_explicit(pr, nr, alb, emi);
        SolveConfig c = exact(base);
        c.method = run.meth;
        c.plane_bias = run.bias;
        c.nee_self_cos = run.self_cos;
        c.soft_eps = 1.0f;
        c.horizon = 0.02f;
        c.ms = run.ms;
        c.no_occlusion = run.no_occ;
        c.budget = uint32_t(pr.size());        // one slice, so one sweep is one pass
        solver.reset(set);
        solver.run_sweeps(set, c, 1, 0);
        const std::vector<glm::vec4> E = set.read_irradiance();

        double sg[16] = {0}, se[16] = {0}, sd[16] = {0};
        int    cnt[16] = {0};
        const double r = double(pr[first_recv].w);
        for (uint32_t k = first_recv; k < uint32_t(pr.size()); ++k) {
            const glm::dvec3 P(pr[k].x, pr[k].y, pr[k].z);
            const double h = P.y / r;
            int b = 0;
            while (b + 1 < nb && h >= edge[b + 1]) ++b;
            const double got = double(E[k].r), ref = exact_E(P, glm::dvec3(1, 0, 0));
            sg[b] += got; se[b] += ref; ++cnt[b];
            if (ref > 1e-9) sd[b] += (got / ref - 1.0) * (got / ref - 1.0);
        }

        std::printf("[GATE]   %-18s", run.tag);
        for (int b = 0; b < nb; ++b) {
            if (cnt[b] == 0) { std::printf("%9s", "-"); continue; }
            const double e = se[b] > 1e-12 ? sg[b] / se[b] - 1.0 : 0.0;
            std::printf("%8.1f%%", 100.0 * e);
            if (edge[b] >= 4.0 && cnt[b] > 8 && !run.no_occ) {
                double& w = run.meth == Method::Radiance ? worst_m1 : worst_m2;
                w = std::max(w, std::fabs(e));
            }
        }
        std::printf("\n");
        // Spread within the bin, which is the "3x the horizontal variation"
        // symptom: a bias moves the mean, a scatter does not.
        std::printf("[GATE]   %-18s", "  rms per-surfel");
        for (int b = 0; b < nb; ++b) {
            if (cnt[b] == 0) { std::printf("%9s", "-"); continue; }
            std::printf("%8.1f%%", 100.0 * std::sqrt(sd[b] / double(cnt[b])));
        }
        std::printf("\n");
    }

    // Away from the crease the corner is an ordinary well-resolved configuration
    // and the kernel has no excuse. M1 meets that.
    report("13 corner  M1 worst bias beyond 4 radii", worst_m1, 0.0, 0.05);

    // M2 does NOT, and the tolerance below is set to catch a regression rather
    // than to certify the number: 5% is the target and the microbuffer is at
    // ~13%. The last row is the evidence for where it goes. `no clamp` is the
    // no-occlusion path, which differs from the shipped one in exactly one way
    // that matters here -- it does not clamp `fill` to 1 -- and it lands within a
    // few percent of the analytic answer at every height. So the deficit is the
    // clamp, and the clamp is not the bug: it is load-bearing in a closed room,
    // where removing it inflates Cornell's converged solve by 31%.
    //
    // Ruled out by measurement, each of these leaving the number unmoved:
    // the depth tolerance (2, 8, 1e4 radii -- byte-identical), the front/behind
    // classification (fill = cov_front instead of cov_any -- byte-identical),
    // bucket count (8x8 vs 16x16), surfel density (8k vs 32k), per-texel solid
    // angle in the deposit and in its normalization, footprint dilation at
    // constant energy, and clipping the normalization sum at the square's rim.
    // Occluder radius inflation (the finding 25 fix, ported to the gather) does
    // seal the crease -- and grows every distant silhouette with it, +15% at 64
    // radii, so it trades one error for a worse one.
    //
    // What is left is the splat's SHAPE, and two experiments say so directly.
    //
    // Coverage is deposited as a circle in TEXEL space, and near the square's rim
    // -- the horizon, which is where a concave crease puts the neighbouring plane
    // -- this map is strongly anisotropic (micro.glsl: 4.2x across the square at
    // 16x16), so a texel-space circle is nothing like the disc's true angular
    // image. Replacing the whole kernel with 32 stratified samples of the disc
    // pushed through the exact forward map -- no kernel, no Jacobian, each sample
    // landing where it actually belongs -- moved the worst bias beyond 4 radii
    // from 13.2% to 9.5%, confirming the shape matters but not closing it.
    //
    // The rest is the packing bound of finding 25 in the gather. Discs of total
    // area A cannot tile an area-A plane, so at grazing incidence they overlap in
    // some directions and leave gaps in others: coverage per bucket is bimodal
    // around a correct mean, which is exactly why the unclamped path is accurate
    // and the clamped one is 20% dark. Dilating the sample disc at CONSTANT
    // energy -- the variance-reduction version of finding 25's radius inflation,
    // which does not grow the silhouette -- took 4-8 radii from 9.5% to 5.4% at a
    // dilation of 2. Neither helped the first two radii, which is the row the
    // render's band is actually on, so neither was kept: both change the
    // estimator everywhere to fix it in one place.
    //
    // A real fix has to make coverage a UNION rather than a sum, the way finding
    // 14 made the NEE visibility a bitmask rather than summed coverage. That is
    // the same lesson for the third time and it is where the next attempt should
    // start.
    report("13 corner  M2 worst bias beyond 4 radii", worst_m2, 0.0, 0.16);
}

// --- gate 6: bake area conservation -----------------------------------------

void gate_bake(SurfelSet& scene, const std::vector<Tri>& tris, uint32_t surfels) {
    scene.build(tris, surfels);
    const double disc = kPi * double(scene.radius()) * double(scene.radius())
                      * double(scene.count());
    report("6 bake  sum(pi r^2) / triangle area", disc / scene.area(), 1.0, 1e-4);
    note("N = %u, r = %.6f, spacing = %.6f, emissive surfels = %u",
         scene.count(), scene.radius(), scene.spacing(), scene.emissive_count());
    if (scene.emissive_count() < 64)
        note("WARNING: fewer than 64 emitter surfels -- the penumbra will band. "
             "Raise SGI_SURFELS.");
}

// --- gate 7: view-transform calibration -------------------------------------

glm::vec3 srgb_oetf(glm::vec3 c) {
    for (int i = 0; i < 3; ++i) {
        float v = std::clamp(c[i], 0.0f, 1.0f);
        c[i] = v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
    }
    return c;
}

void gate_tonemap() {
    // The Cornell panel's radiance, and what the reference renders it as.
    const glm::vec3 Le(17.0f, 17.0f * 0.70588f, 17.0f * 0.23529f);
    const glm::vec3 ref(254.0f, 250.0f, 238.0f);

    auto show = [&](const char* name, glm::vec3 v) {
        v = srgb_oetf(v) * 255.0f;
        note("%-10s (17.0, 12.0, 4.0) -> (%.0f, %.0f, %.0f)   reference (254, 250, 238)",
             name, double(v.r), double(v.g), double(v.b));
    };
    auto aces = [](glm::vec3 x) {
        const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
        return glm::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
    };
    show("ACES", aces(Le));
    show("Reinhard", Le / (1.0f + Le));
    show("Clamp", glm::clamp(Le, 0.0f, 1.0f));

    // Filmic (Hejl & Burgess-Dawson) carries its own encode, so no OETF.
    glm::vec3 t = glm::max(glm::vec3(0.0f), Le - 0.004f);
    glm::vec3 f = (t * (6.2f * t + 0.5f)) / (t * (6.2f * t + 1.7f) + 0.06f);
    f = glm::clamp(f, 0.0f, 1.0f) * 255.0f;
    note("%-10s (17.0, 12.0, 4.0) -> (%.0f, %.0f, %.0f)   reference (254, 250, 238)",
         "Filmic", double(f.r), double(f.g), double(f.b));

    note("Both references are Blender Cycles renders. A saturated (17,12,4) landing");
    note("near-neutral at (254,250,238) WITHOUT clipping is Blender 4.x AgX, which");
    note("none of the above is. The numeric diff is approximate until AgX is ported;");
    note("pick whichever transform lands closest and record which one in the log.");
    (void)ref;
}

} // namespace

// --- gate 11: the NEE partition, against an analytic rectangle ---------------
//
// Two questions in one gate, and they fail differently:
//
//   * is the UNOCCLUDED direct term right? A rectangular emitter above a point
//     has a closed-form irradiance, so the 64-bit quadrature can be checked
//     against it exactly. Getting this wrong shows up as a global brightness
//     error that is easy to mistake for a shading bug somewhere else.
//   * does the PARTITION hold? With the emission removed from lout and added
//     back by the direct pass, the total must not move. Anything else is double
//     counting or a dropped term.
//
// The receiver is a single surfel on the axis, so no surfel-density or coverage
// effect enters and the number is the transport alone.
void gate_partition(SurfelSet& set, Solver& solver, const SolveConfig& base) {
    const float hu = 0.235f, hv = 0.19f;    // Cornell's panel, half-extents
    const float h  = 1.98f;                 // receiver distance below it
    const float L  = 3.0f;

    // The emitter as two triangles, for the proxy fit.
    const glm::vec3 c0(-hu, h, -hv), c1(hu, h, -hv), c2(hu, h, hv), c3(-hu, h, hv);
    std::vector<Tri> tris(2);
    tris[0].p[0] = c0; tris[0].p[1] = c1; tris[0].p[2] = c2;
    tris[1].p[0] = c0; tris[1].p[1] = c2; tris[1].p[2] = c3;
    for (Tri& t : tris) {
        t.n = glm::vec3(0, -1, 0);
        t.emission = glm::vec3(L);
        t.albedo = glm::vec3(0.0f);
        t.area = 0.5f * glm::length(glm::cross(t.p[1] - t.p[0], t.p[2] - t.p[0]));
    }

    // Closed form for a point below a rectangle, by the standard decomposition
    // into four corner quadrants. The bracket is the corner form factor times
    // 2*pi:
    //   F_corner = (1/2pi) [ a/sqrt(a^2+d^2) atan(b/sqrt(a^2+d^2)) + (a<->b) ]
    //
    // and then E = pi * L * F, because F is defined with the 1/pi in it:
    // F = (1/pi) INTEGRAL cos dOmega, while E = L * INTEGRAL cos dOmega.
    //
    // Writing E = L * F here instead -- dropping that pi -- is what the first
    // version of this gate did, and it reported both the NEE path and the
    // microbuffer path as 217% high. Two independent implementations agreeing
    // with each other and disagreeing with the analytic answer is the signature
    // of a wrong analytic answer, which is section 9's whole point about being
    // "off by pi".
    auto corner = [&](double a, double b, double d) {
        const double ra = std::sqrt(a * a + d * d), rb = std::sqrt(b * b + d * d);
        return (a / ra) * std::atan(b / ra) + (b / rb) * std::atan(a / rb);
    };
    const double analytic = 2.0 * double(L) * corner(double(hu), double(hv), double(h));

    // The emissive surfels, so the microbuffer path has something to see, plus
    // one receiver on the axis.
    const uint32_t m = 4096;
    std::vector<glm::vec4> pr; std::vector<glm::vec3> nr, alb, emi;
    const uint32_t side = 64;
    const float cell_u = 2.0f * hu / float(side), cell_v = 2.0f * hv / float(side);
    const float r_e = std::sqrt(cell_u * cell_v / float(kPi));
    for (uint32_t j = 0; j < side; ++j)
        for (uint32_t i = 0; i < side; ++i) {
            pr.push_back(glm::vec4(-hu + (float(i) + 0.5f) * cell_u, h,
                                   -hv + (float(j) + 0.5f) * cell_v, r_e));
            nr.push_back(glm::vec3(0, -1, 0));
            alb.push_back(glm::vec3(0.0f));
            emi.push_back(glm::vec3(L));
        }
    (void)m;
    const uint32_t receiver = uint32_t(pr.size());
    pr.push_back(glm::vec4(0.0f, 0.0f, 0.0f, 0.01f));
    nr.push_back(glm::vec3(0, 1, 0));
    alb.push_back(glm::vec3(0.0f));
    emi.push_back(glm::vec3(0.0f));

    set.build_explicit(pr, nr, alb, emi);

    SurfelGrid grid;
    grid.build(set, 1.0f);
    EmitterSet em;
    em.build(tris);
    report("11 partition  proxy area vs source",
           em.proxy_area(), em.source_area(), 1e-3);

    SolveConfig c = exact(base);
    c.method = Method::Micro;

    // NEE off: the microbuffer alone.
    solver.attach(nullptr, nullptr);
    solver.reset(set);
    c.nee = false;
    solver.run_sweeps(set, c, 1, 0);
    const double e_micro = double(set.read_irradiance()[receiver].r);

    // NEE on: emission out of lout, direct added back explicitly.
    solver.attach(&grid, &em);
    solver.reset(set);
    c.nee = true;
    solver.run_sweeps(set, c, 1, 0);
    const double e_nee = double(set.read_irradiance()[receiver].r);
    solver.attach(nullptr, nullptr);

    std::printf("[GATE]   rectangle above a point: analytic %.6f, "
                "NEE %.6f (%+.2f%%), microbuffer %.6f (%+.2f%%)\n",
                analytic, e_nee, 100.0 * (e_nee / analytic - 1.0),
                e_micro, 100.0 * (e_micro / analytic - 1.0));
    // 8x8 over a smooth integrand: the quadrature error is small and one-sided.
    report("11 partition  NEE vs analytic rect", e_nee, analytic, 0.02);
}

bool run_gates(const std::string& which, SurfelSet& scene, Solver& solver,
               const SolveConfig& base, const std::vector<Tri>& tris) {
    g_pass = g_fail = 0;
    const bool all = which == "all" || which == "1";
    auto want = [&](const char* n) {
        return all || which.find(n) != std::string::npos;
    };

    const uint32_t gate_surfels = 8000;   // fast; the gates test math, not density

    // NEE needs emitter proxies FITTED TO THE CURRENT SCENE, and every gate below
    // replaces the surfel set with an explicit one of its own. Leaving the
    // scene's proxies attached would remove the emission from lout (so M1 reads
    // zero) and add back a direct term computed against a grid and a panel that
    // belong to a different set. Detach for the duration; the partition is gated
    // separately by gate 11, which fits proxies to its own scene.
    solver.attach(nullptr, nullptr);

    std::printf("[GATE] running: %s\n", which.c_str());
    if (want("bake"))    gate_bake(scene, tris, gate_surfels);
    if (want("tonemap")) gate_tonemap();
    if (want("p2p"))     gate_p2p(scene, solver, base);
    if (want("disc"))    gate_disc(scene, solver, base);
    if (want("hemi"))    gate_hemi(scene, solver, base);
    if (want("scale"))   gate_scale(scene, solver, base, tris, gate_surfels);
    if (want("noocc"))   gate_noocc(scene, solver, base, tris, gate_surfels);
    if (want("opaque")) gate_opaque(scene, solver, base);
    if (want("partition")) gate_partition(scene, solver, base);
    if (want("graze")) gate_graze(scene, solver, base);
    if (want("corner")) gate_corner(scene, solver, base);
    if (want("occ")) {
        // Honours SGI_SURFELS and SGI_BUCKETS so the bias can be swept against
        // both: a stochastic-coverage bias falls as either rises, a structural
        // cull does not.
        const char* sv = std::getenv("SGI_SURFELS");
        gate_occ(scene, solver, base, tris, sv ? uint32_t(std::atoi(sv)) : 30000u);
    }

    std::printf("[GATE] %d passed, %d failed\n", g_pass, g_fail);
    std::fflush(stdout);
    return g_fail == 0;
}

} // namespace sgi

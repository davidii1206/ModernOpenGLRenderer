// ---------------------------------------------------------------------------
// Example 41 — multi-bounce G-buffer GI, brute-force reference
//
// multi-bounce-gbuffer.md, variant A (the recursive N^3 formulation), with every
// acceleration in the document deliberately left out.
//
// The idea being tested: for each shading point in the camera's G-buffer, put a
// secondary camera there, point it along the normal, and RASTERIZE the scene
// from it. The resulting tiny image is the incoming radiance over that point's
// hemisphere; integrate it and you have that point's indirect lighting, with
// off-screen and back-facing geometry included -- which is what separates this
// from screen-space GI. Do it again at the hit points and you have another
// bounce.
//
// What this example is:
//
//   - one secondary camera per GI-grid pixel (doc section 4.1's "full-resolution
//     naive version ... a correctness reference, not a shipping configuration")
//   - a hemi-octahedral target (4.3) rasterized in compute, one workgroup per
//     camera, visibility resolved by atomicMin in shared memory (5.3)
//   - the full-resolution mesh: every camera loops every triangle, no culling,
//     no LOD, no cluster DAG, no work list (5.2, 5.4 are what this measures)
//   - recursion to MBG_BOUNCES levels with the resolution ladder and tile
//     clustering of 6.2, terminating in direct lighting only (6.1)
//   - the direct term as a separate pass (3): exact polygon irradiance, with
//     visibility taken off the rasterized depth sort -- per camera for the
//     bounce transport, and a hemisphere per PIXEL for the image, which is the
//     only way its shadow boundaries come out sharp
//
// What it is NOT: variant B. There is no radiance cache and nothing reads last
// frame's output, so a completed sweep is correct on its own terms rather than
// converging toward correct -- which is the property that makes it usable as the
// reference variant B gets validated against (7.4, 8.2 milestone 7).
//
// The external gate is the same pair of path-traced images example 40 uses, at
// the same camera:
//   CornellBoxGroundTruthDirectLighting.png   MBG_BOUNCES=1
//   CornellBoxOriginalGroundTruth.png         MBG_BOUNCES=3 and up
//
// Run from this target's build directory: the model, shaders/ and both reference
// PNGs are resolved relative to the working directory.
//
// Every knob is an env var as well as an ImGui control, so a measurement can be
// scripted. The ones that matter, with defaults:
//
//   MBG_DAYLIGHT=1          a sun and a sky dome through the box's open side;
//                           off by default, so nothing above it changes
//   MBG_SKY=f               constant radiance added in every direction
//   MBG_SKY_ZENITH=f|r,g,b  the dome: straight up, at the horizon, and below it
//   MBG_SKY_HORIZON=..      (MBG_DAYLIGHT sets all three; these override it)
//   MBG_SKY_GROUND=..
//   MBG_SKY_UP=x,y,z        world up for the dome's gradient; glTF is Y-up
//   MBG_SUN=f|r,g,b         irradiance on a surface facing the sun
//   MBG_SUN_DIR=x,y,z       toward the sun
//   MBG_SUN_ANGLE=deg       the sun's angular RADIUS; larger is a softer shadow
//   MBG_GATE=all            run the analytic gates and exit
//   MBG_MODEL=path.glb      CornellBoxOriginal.glb; the references only match it
//   MBG_BOUNCES=3           camera levels; 1 == direct only
//   MBG_NEE=1               analytic direct term (doc section 3's separate pass)
//   MBG_TENT=1              spread each texel's mass over the 4 nearest tiles
//   MBG_DIRECT_PIXEL=1      rasterize a hemisphere per pixel for the image's
//                           direct term, not one per GI-grid camera
//   MBG_DIRECT_RES=16       edge of the per-pixel pass's light view
//   MBG_LV_RES=8            edge of the secondary cameras' light view
//   MBG_INDIRECT_ONLY=1     composite the bounce term alone, for inspecting it
//   MBG_JITTER=1            rotate each receiver's tangent frame (decorrelate)
//   MBG_FILTER=3            a-trous denoise iterations over the GI grid
//   MBG_FILTER_R=2          taps per side for that filter
//   MBG_SCALE=4             GI grid = framebuffer / scale; 1 == one camera/pixel
//   MBG_BUDGET=4096         level-1 cameras per frame
//   MBG_RES=32              level-1 target edge; MBG_RES2/3/4 for deeper levels
//   MBG_BLOCK=8             spawn tile edge; 1 == a child per texel
//   MBG_GTCAM=0|1|2         free | reference camera at 512^2 | at 1600x900
//   MBG_SOLVE=n             complete n sweeps before the first present, then hold
//   MBG_COMPARE=0|1         print RMSE against the reference after the solve
//   MBG_BENCH=n             n frames, print per-pass timings, exit
//   MBG_SHOT=path.png       write the final frame
//   MBG_NOGUI=1             no ImGui -- required for a clean screenshot
// ---------------------------------------------------------------------------

#include "gpu_util.hpp"
#include "scene.hpp"
#include "screen.hpp"
#include "solver.hpp"
#include "validate.hpp"

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>
#include <gllib/log.hpp>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace mbg;

namespace {

constexpr const char* kModelPath = "CornellBoxOriginal.glb";

// Camera matched to the two path-traced references. Derived in example 40 from
// the reference images themselves (the back wall spans 57.3% of the frame at row
// 256, and the walls are 2.02 units apart); copied here unchanged so the two
// examples' screenshots are pixel-comparable to each other as well as to the
// references. Native glb units: floor y = 0, ceiling y = 1.99, back wall
// z = -1.04, opening toward +z.
constexpr glm::vec3 kGtEye{0.004f, 0.999f, 3.864f};
constexpr glm::vec3 kGtTarget{0.004f, 0.999f, 0.0f};
constexpr float     kGtFovY = 38.75f;
constexpr int       kGtRes  = 512;

const char* const kTonemapNames[] = {"ACES", "Reinhard", "Clamp", "Filmic", "AgX"};
constexpr int kTonemapCount = 5;
// Reinhard, on measurement rather than on principle. Example 40 reasoned that
// both references must be AgX renders (the panel lands near-neutral without
// clipping) and called an AgX port the follow-up; this example has that port,
// and per-patch it is not the closest match. Against the direct reference's back
// wall, reference (112,91,47): Reinhard (105,90,53), ACES (134,109,47), AgX
// (112,97,84) -- AgX gets the red and green almost exactly and then desaturates
// the blue to nearly double. Either the renders are not AgX, or the standard
// approximation of it diverges from Blender's OCIO transform at this saturation.
//
// The tone curve is a presentation choice, so the default is the one that
// measures closest and all five stay available. It matters more than it sounds:
// at this point the curve contributes as much to the RMSE as the transport does,
// which is why implementation.md reports patches and not just a single number.
constexpr int kTonemapDefault = 1;

struct EnvOpts {
    std::string gate;
    std::string model = kModelPath;
    SolveConfig cfg;
    int   gtcam = 0;
    int   view = 7;
    int   gt_index = 1;
    uint32_t solve = 0;
    int   bench = 0;
    bool  shot = false;
    std::string shot_path;
    bool  nogui = false;
    bool  compare = false;
    int   tonemap = kTonemapDefault;
    float exposure = 1.0f;
    bool  paused = false;
};

// "0.4" -> (0.4, 0.4, 0.4); "0.4,0.5,0.6" -> the triple. One spelling for a grey
// and one for a colour, because most of these knobs are set to a grey while
// being measured and to a colour while being looked at.
glm::vec3 parse_vec3(const char* v, const glm::vec3& fallback) {
    float a = 0.0f, b = 0.0f, c = 0.0f;
    int n = sscanf(v, "%f,%f,%f", &a, &b, &c);
    if (n == 3) return glm::vec3(a, b, c);
    if (n == 1) return glm::vec3(a);
    return fallback;
}

EnvOpts read_env() {
    EnvOpts o;
    auto u32 = [](const char* v, uint32_t& dst) { dst = uint32_t(std::max(0, atoi(v))); };
    if (const char* v = getenv("MBG_GATE"))     o.gate = v;
    if (const char* v = getenv("MBG_MODEL"))    o.model = v;
    if (const char* v = getenv("MBG_BOUNCES"))  u32(v, o.cfg.bounces);
    if (const char* v = getenv("MBG_SCALE"))    u32(v, o.cfg.scale);
    if (const char* v = getenv("MBG_BUDGET"))   u32(v, o.cfg.budget);
    if (const char* v = getenv("MBG_RES"))      u32(v, o.cfg.res[0]);
    if (const char* v = getenv("MBG_RES2"))     u32(v, o.cfg.res[1]);
    if (const char* v = getenv("MBG_RES3"))     u32(v, o.cfg.res[2]);
    if (const char* v = getenv("MBG_RES4"))     u32(v, o.cfg.res[3]);
    if (const char* v = getenv("MBG_BLOCK"))    u32(v, o.cfg.block[0]);
    if (const char* v = getenv("MBG_BLOCK2"))   u32(v, o.cfg.block[1]);
    if (const char* v = getenv("MBG_BLOCK3"))   u32(v, o.cfg.block[2]);
    if (const char* v = getenv("MBG_BIAS"))     o.cfg.bias = float(atof(v));
    // The preset first, so the individual knobs below can override any part of it.
    if (const char* v = getenv("MBG_DAYLIGHT")) { if (atoi(v) != 0) o.cfg.sky = SkyLight::daylight(); }
    if (const char* v = getenv("MBG_SKY"))      o.cfg.sky.ambient = parse_vec3(v, o.cfg.sky.ambient);
    if (const char* v = getenv("MBG_SKY_ZENITH"))  o.cfg.sky.zenith  = parse_vec3(v, o.cfg.sky.zenith);
    if (const char* v = getenv("MBG_SKY_HORIZON")) o.cfg.sky.horizon = parse_vec3(v, o.cfg.sky.horizon);
    if (const char* v = getenv("MBG_SKY_GROUND"))  o.cfg.sky.ground  = parse_vec3(v, o.cfg.sky.ground);
    if (const char* v = getenv("MBG_SKY_UP"))      o.cfg.sky.up      = parse_vec3(v, o.cfg.sky.up);
    if (const char* v = getenv("MBG_SUN"))         o.cfg.sky.irradiance = parse_vec3(v, o.cfg.sky.irradiance);
    if (const char* v = getenv("MBG_SUN_DIR"))     o.cfg.sky.dir     = parse_vec3(v, o.cfg.sky.dir);
    if (const char* v = getenv("MBG_SUN_ANGLE"))   o.cfg.sky.angle   = float(atof(v));
    if (const char* v = getenv("MBG_EMISSIVE")) o.cfg.emissive = float(atof(v));
    if (const char* v = getenv("MBG_TWOSIDED")) o.cfg.two_sided = atoi(v) != 0;
    if (const char* v = getenv("MBG_NEE"))      o.cfg.nee = atoi(v) != 0;
    if (const char* v = getenv("MBG_TENT"))     o.cfg.tent = atoi(v) != 0;
    if (const char* v = getenv("MBG_DIRECT_PIXEL")) o.cfg.direct_pixel = atoi(v) != 0;
    if (const char* v = getenv("MBG_DIRECT_RES")) u32(v, o.cfg.direct_res);
    if (const char* v = getenv("MBG_LV_RES"))    u32(v, o.cfg.cam_lv_res);
    if (const char* v = getenv("MBG_INDIRECT_ONLY")) o.cfg.indirect_only = atoi(v) != 0;
    if (const char* v = getenv("MBG_JITTER"))   o.cfg.jitter = atoi(v) != 0;
    if (const char* v = getenv("MBG_FILTER"))   u32(v, o.cfg.filter_iters);
    if (const char* v = getenv("MBG_FILTER_R")) o.cfg.filter_radius = std::max(1, atoi(v));
    if (const char* v = getenv("MBG_PLANE"))    o.cfg.plane_tol = float(atof(v));
    if (const char* v = getenv("MBG_GTCAM"))    o.gtcam = atoi(v);
    if (const char* v = getenv("MBG_VIEW"))     o.view = atoi(v);
    if (const char* v = getenv("MBG_REF"))      o.gt_index = atoi(v);
    if (const char* v = getenv("MBG_SOLVE"))    u32(v, o.solve);
    if (const char* v = getenv("MBG_BENCH"))    o.bench = atoi(v);
    if (const char* v = getenv("MBG_SHOT"))     { o.shot = true; o.shot_path = v; }
    if (const char* v = getenv("MBG_NOGUI"))    o.nogui = atoi(v) != 0;
    if (const char* v = getenv("MBG_COMPARE"))  o.compare = atoi(v) != 0;
    if (const char* v = getenv("MBG_TONEMAP"))  o.tonemap = atoi(v);
    if (const char* v = getenv("MBG_EXPOSURE")) o.exposure = float(atof(v));
    if (const char* v = getenv("MBG_PAUSE"))    o.paused = atoi(v) != 0;
    return o;
}

// Percentiles over the steady-state tail. The MINIMUM of the tail is the least
// clock-perturbed sample, so it is the number to compare across runs; the spread
// says how noisy the machine was.
void report_bench(std::vector<double> ms) {
    if (ms.empty()) return;
    ms.erase(ms.begin(), ms.begin() + ms.size() / 3);
    if (ms.empty()) return;
    std::sort(ms.begin(), ms.end());
    auto pct = [&](double p) { return ms[std::size_t(p * double(ms.size() - 1))]; };
    printf("[bench] n=%zu  min=%.3f  p10=%.3f  p50=%.3f  p90=%.3f ms\n",
           ms.size(), ms.front(), pct(0.10), pct(0.50), pct(0.90));
}

// Numeric comparison against a reference PNG, in DISPLAY space.
//
// Both are already tonemapped -- the references by Blender, ours by display.frag
// -- so this compares what a viewer sees. It is the honest place to compare and
// also the lossy one: none of our tone curves is AgX, so a few percent of the
// error reported here is the curve rather than the transport. Example 40
// documents the same gap. Sizes must match, which is what MBG_GTCAM=1 is for.
void compare_to_reference(const References& refs, int gt_index, int w, int h) {
    const gfx::Texture& ref = gt_index == 0 ? refs.direct : refs.full;
    const bool have = gt_index == 0 ? refs.have_direct : refs.have_full;
    if (!have) { gllib::log(gllib::LogLevel::warn, "compare: reference not loaded"); return; }
    if (ref.width() != w || ref.height() != h) {
        gllib::logf(gllib::LogLevel::warn,
                    "compare: reference is %dx%d but the framebuffer is %dx%d "
                    "-- run with MBG_GTCAM=1", ref.width(), ref.height(), w, h);
        return;
    }

    std::vector<unsigned char> ours(std::size_t(w) * h * 4);
    std::vector<unsigned char> theirs(std::size_t(w) * h * 4);
    glReadBuffer(GL_BACK);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, ours.data());
    glGetTextureImage(ref.handle(), 0, GL_RGBA, GL_UNSIGNED_BYTE,
                      GLsizei(theirs.size()), theirs.data());

    // glReadPixels is bottom-up, stb_image loaded the PNG top-down. Flip ours
    // once here so everything below indexes the same way the reference does.
    std::vector<unsigned char> flipped(ours.size());
    for (int y = 0; y < h; ++y)
        std::memcpy(&flipped[std::size_t(y) * w * 4],
                    &ours[std::size_t(h - 1 - y) * w * 4], std::size_t(w) * 4);

    double se = 0.0, ae = 0.0, peak = 0.0;
    std::size_t n = 0;
    for (std::size_t i = 0; i < flipped.size(); ++i) {
        if ((i & 3) == 3) continue;                 // alpha
        const double d = (double(flipped[i]) - double(theirs[i])) / 255.0;
        se += d * d;
        ae += std::abs(d);
        peak = std::max(peak, std::abs(d));
        ++n;
    }
    const double rmse = std::sqrt(se / double(std::max<std::size_t>(1, n)));
    printf("[compare] vs %s: RMSE %.4f  MAE %.4f  peak %.4f  (display space, 0..1)\n",
           gt_index == 0 ? "CornellBoxGroundTruthDirectLighting.png"
                         : "CornellBoxOriginalGroundTruth.png",
           rmse, ae / double(std::max<std::size_t>(1, n)), peak);

    // --- Roughness ----------------------------------------------------------
    //
    // RMSE cannot see this example's artifacts. Measured against the direct
    // reference, the analytic direct term and the quadrature one score the SAME
    // 0.0468 -- while one image is smooth and the other is covered in mottle.
    // Two reasons: a high-frequency error averages to almost nothing in a
    // per-pixel mean, and the residual is dominated by the tone curve (finding
    // 8), which is a large smooth offset that drowns everything else.
    //
    // So the artifacts get their own instrument, the one example 40 used on its
    // reconstruction: high-pass energy over patches that are provably smooth in
    // the reference. Each pixel minus the mean of its 5x5 neighbourhood, RMS over
    // the patch, in 0..255 units. The reference is a converged path trace, so its
    // value is the noise floor of the comparison; ours above that is artifact.
    // Patches chosen to be smooth AND shadow-free in the reference: a shadow
    // boundary is detail, not roughness, and including one would reward blur.
    struct Patch { const char* name; int x, y, w, h; };
    static const Patch patches[] = {
        {"back wall",  140, 110, 230,  80},
        {"left wall",   15, 150,  55, 180},
        {"right wall", 450, 150,  50, 180},
        {"ceiling",    110,  15,  80,  45},
    };
    // 15x15, not 5x5. The artifacts this is meant to see live at the GI grid's
    // scale -- 8 pixels at MBG_SCALE=8 -- and a 5x5 high-pass looks straight past
    // them: measured on the same pair of images it reports 1.06 against 0.54
    // where the 15x15 reports 3.93 against 1.31.
    constexpr int kKernel = 15;
    auto roughness = [&](const std::vector<unsigned char>& img, const Patch& p) {
        double acc = 0.0;
        int count = 0;
        constexpr int r = kKernel / 2;
        for (int y = std::max(p.y, r); y < std::min(p.y + p.h, h - r); ++y) {
            for (int x = std::max(p.x, r); x < std::min(p.x + p.w, w - r); ++x) {
                for (int c = 0; c < 3; ++c) {
                    double mean = 0.0;
                    for (int dy = -r; dy <= r; ++dy)
                        for (int dx = -r; dx <= r; ++dx)
                            mean += double(img[(std::size_t(y + dy) * w + x + dx) * 4 + c]);
                    mean /= double(kKernel * kKernel);
                    const double d = double(img[(std::size_t(y) * w + x) * 4 + c]) - mean;
                    acc += d * d;
                    ++count;
                }
            }
        }
        return count ? std::sqrt(acc / double(count)) : 0.0;
    };
    printf("[compare] roughness (15x15 high-pass RMS, 0..255; the reference is a "
           "converged path trace, so its column is the noise floor)\n");
    double sum_ours = 0.0, sum_ref = 0.0;
    for (const Patch& p : patches) {
        const double a = roughness(flipped, p);
        const double b = roughness(theirs, p);
        sum_ours += a;
        sum_ref += b;
        printf("[compare]   %-11s ours %6.3f   reference %6.3f   %5.2fx\n",
               p.name, a, b, b > 1e-6 ? a / b : 0.0);
    }
    const double np = double(sizeof(patches) / sizeof(patches[0]));
    printf("[compare]   %-11s ours %6.3f   reference %6.3f   %5.2fx\n", "MEAN",
           sum_ours / np, sum_ref / np, sum_ref > 1e-6 ? sum_ours / sum_ref : 0.0);
}

} // namespace

int main() {
    gllib::log_to_stderr(gllib::LogLevel::info);
    const EnvOpts env = read_env();

    gfx::WindowDesc wd;
    wd.title = "41 — multi-bounce G-buffer, brute force";
    // The reference PNGs are 512x512. Matching the framebuffer exactly makes the
    // split, the diff and MBG_COMPARE pixel-aligned against them.
    wd.width  = env.gtcam == 1 ? kGtRes : 1600;
    wd.height = env.gtcam == 1 ? kGtRes : 900;
    wd.vsync = false;
    wd.debug = true;
    gfx::Window window(wd);
    window.vsync(false);
    gl::enable_debug_output(false);

    gfx::ImGuiOverlay gui;
    if (!env.nogui) gui.init(window);

    // --- Scene ---------------------------------------------------------------

    auto model = std::make_unique<gfx::Model>();
    if (!model->load(env.model.c_str())) {
        gllib::logf(gllib::LogLevel::error, "failed to load '%s'", env.model.c_str());
        return 1;
    }

    const std::vector<Tri> tris = extract_triangles(*model);
    if (tris.empty()) {
        gllib::log(gllib::LogLevel::error, "no triangles extracted");
        return 1;
    }
    Scene scene;
    if (!scene.build(tris)) return 1;

    const Bounds& sb = scene.bounds();
    gllib::logf(gllib::LogLevel::info,
                "scene: %u tris (%u emissive), area %.4f, "
                "bounds [%.2f %.2f %.2f]..[%.2f %.2f %.2f], diagonal %.3f",
                scene.count(), scene.emissive_count(), scene.area(),
                sb.mn.x, sb.mn.y, sb.mn.z, sb.mx.x, sb.mx.y, sb.mx.z, sb.diagonal());

    // --- Passes --------------------------------------------------------------

    GBuffer gbuf;
    gbuf.create(window.framebuffer_width(), window.framebuffer_height());

    GeometryPass geometry;
    DisplayPass display;
    Solver solver;
    if (!geometry.init() || !display.init() || !solver.init()) {
        gllib::log(gllib::LogLevel::error, "shader initialisation failed");
        return 1;
    }

    References refs;
    refs.load();

    SolveConfig cfg = env.cfg;
    // The camera sits ON a surface, so its own triangle passes through the
    // origin of its hemisphere. Scale the push-off with the scene rather than
    // hard-coding a number that is fine for a 2 m Cornell box and catastrophic
    // for a 200 m one.
    if (getenv("MBG_BIAS") == nullptr) cfg.bias = sb.diagonal() * 2.5e-4f;
    if (getenv("MBG_PLANE") == nullptr) cfg.plane_tol = sb.diagonal() * 0.01f;
    cfg.running = !env.paused;

    {
        Quadrature q;
        q.build(cfg.res[0]);
        gllib::logf(gllib::LogLevel::info,
                    "quadrature %ux%u: sum(dOmega) = %.7f (2pi = %.7f), "
                    "sum(cos dOmega) = %.7f (pi = %.7f)",
                    cfg.res[0], cfg.res[0], q.sum_omega, 2.0 * 3.14159265358979,
                    q.sum_cos, 3.14159265358979);
    }

    // --- Gates ---------------------------------------------------------------
    //
    // They need a GL context but not a frame, so they run here and the process
    // exits. A PI error is invisible in an image and consistent across near and
    // far, so it can only be caught against an analytic answer.
    if (!env.gate.empty()) {
        const bool ok = run_gates(env.gate, solver, cfg, scene, tris);
        if (!env.nogui) gui.shutdown();
        return ok ? 0 : 1;
    }

    // --- Camera --------------------------------------------------------------

    gfx::Camera cam;
    const float radius = std::max(0.1f, sb.radius());
    cam.perspective(env.gtcam != 0 ? kGtFovY : 45.0f,
                    float(window.framebuffer_width()) /
                        float(std::max(1, window.framebuffer_height())),
                    radius * 0.002f, radius * 20.0f);
    if (env.gtcam != 0) {
        cam.look_at(kGtEye, kGtTarget);
    } else {
        cam.look_at(sb.center() + glm::vec3(0.0f, 0.0f, radius * 2.2f), sb.center());
    }

    solver.configure(cfg, gbuf.width, gbuf.height);
    gllib::logf(gllib::LogLevel::info,
                "GI grid %dx%d (%u cameras/sweep), %u level(s), %.1f MB of camera state",
                solver.gi_size().x, solver.gi_size().y, solver.gi_pixels(),
                solver.levels(), double(solver.bytes()) / (1024.0 * 1024.0));
    for (uint32_t l = 0; l < solver.levels(); ++l) {
        const LevelInfo& li = solver.level(l);
        gllib::logf(gllib::LogLevel::info,
                    "  level %u: %ux%u target, tile %u -> %u children, "
                    "%u cameras/chunk", l + 1, li.res, li.res, li.block, li.children,
                    li.cameras);
    }

    // --- Timers --------------------------------------------------------------

    PassTimer t_frame("Frame", false);
    PassTimer t_gbuf("G-buffer");
    PassTimer t_display("Display");
    PassTimer t_imgui("ImGui");
    PassTimer* const timers[] = {&t_frame, &t_gbuf, &t_display, &t_imgui};
    PassTimer* const solver_timers[] = {&solver.t_place(), &solver.t_raster(0),
                                        &solver.t_raster(1), &solver.t_raster(2),
                                        &solver.t_raster(3), &solver.t_gather(),
                                        &solver.t_upsample(), &solver.t_direct(),
                                        &solver.t_filter()};

    // --- State ---------------------------------------------------------------

    int view_mode = env.view;
    int gt_index = std::clamp(env.gt_index, 0, 1);
    int tonemap = std::clamp(env.tonemap, 0, kTonemapCount - 1);
    float exposure = env.exposure;
    float irradiance_gain = 1.0f;
    float diff_gain = 4.0f;
    float split_x = 0.5f;
    bool captured = false;
    bool compare_done = false;
    int frame_index = 0;
    bool shot_done = false;
    double last_time = window.time();
    double window_accum = 0.0;
    std::vector<double> bench_ms;
    glm::mat4 last_view_proj = cam.view_projection();

    // Scripted shots want a finished sweep, not a progressive one -- and then
    // they want it to STOP, so that however many frames the shot takes, the image
    // is exactly the requested solve and not that plus whatever the budget got
    // through afterwards.
    uint32_t presolve = env.solve;

    while (!window.should_close()) {
        const double now = window.time();
        const double frame_ms = (now - last_time) * 1000.0;
        const float dt = float(std::min(now - last_time, 0.1));
        last_time = now;
        // Wall clock, not the CPU span of the loop: with vsync off and no sync
        // point the CPU runs far ahead and would report submission cost.
        if (frame_index > 0) t_frame.submit_external(frame_ms);

        window.poll_events();

        const int fw = window.framebuffer_width();
        const int fh = window.framebuffer_height();
        if (fw > 0 && fh > 0 && (fw != gbuf.width || fh != gbuf.height)) {
            gbuf.create(fw, fh);
            cam.set_aspect(float(fw) / float(fh));
        }

        camera_control(window, cam, dt, !env.nogui && !gui.wants_mouse(), captured);

        geometry.poll();
        display.poll();
        solver.poll();

        const glm::mat4 view_proj = cam.view_projection();
        // Every camera in the solve is placed from THIS frame's G-buffer, so a
        // moved camera invalidates the whole sweep rather than part of it. The
        // image keeps its old contents while the new sweep fills in -- stale
        // lighting on moving geometry, which is honest for a reference renderer
        // and is the one place this variant behaves like a progressive one.
        if (view_proj != last_view_proj) {
            solver.restart();
            last_view_proj = view_proj;
        }

        // 1. Primary G-buffer, the one hardware raster pass (doc section 5.5).
        {
            ScopedPass p(t_gbuf);
            geometry.render(gbuf, *model, view_proj);
        }

        // 2. One chunk of the recursive solve.
        solver.configure(cfg, gbuf.width, gbuf.height);
        if (presolve > 0) {
            // Run whole sweeps up front. Each sweep is ceil(pixels/budget) chunks.
            const uint32_t chunks =
                (solver.gi_pixels() + solver.level(0).cameras - 1) / solver.level(0).cameras;
            SolveConfig run = cfg;
            run.running = true;
            solver.restart();
            // glFinish on both sides, so this measures the solve rather than how
            // long it took to SUBMIT the solve. This is the honest cost number
            // for the technique: one complete sweep is one finished image, where
            // a frame is an arbitrary slice of one.
            glFinish();
            const double t0 = window.time();
            for (uint32_t s = 0; s < presolve; ++s)
                for (uint32_t c = 0; c < chunks; ++c) solver.step(gbuf, cam, scene, run);
            glFinish();
            const double sweep_ms = (window.time() - t0) * 1000.0 / double(presolve);
            gllib::logf(gllib::LogLevel::info,
                        "pre-solved %u sweep(s), holding", presolve);
            printf("[sweep] %.1f ms per sweep  (%u chunks, %.0f cameras, %.4g texels, "
                   "%.4g triangle-rasters)\n",
                   sweep_ms, chunks, solver.sweep_cameras(), solver.sweep_texels(),
                   solver.sweep_cameras() * double(scene.count()));
            presolve = 0;
            cfg.running = false;
        } else {
            solver.step(gbuf, cam, scene, cfg);
        }
        // Denoise the grid before it is upsampled. Display only: the solve is
        // still iterating on the unfiltered values, so transport and every gate
        // are untouched -- the same split example 40 makes.
        solver.filter(gbuf, cam, cfg);
        solver.upsample(gbuf, cam, cfg);
        // The image's direct term, per pixel, composited onto the upsampled
        // indirect. Outside the pass above and with its own timer: PassTimer
        // wraps a GL query object, and a query cannot begin while another is
        // active.
        solver.direct_pixel(gbuf, cam, scene, cfg);

        // 3. Display.
        {
            ScopedPass p(t_display);
            DisplayPass::Params dp;
            dp.view_mode = view_mode;
            dp.exposure = exposure;
            dp.irradiance_gain = irradiance_gain;
            dp.diff_gain = diff_gain;
            dp.split_x = split_x;
            dp.gt_index = gt_index;
            dp.tonemap = tonemap;
            dp.inv_view_proj = glm::inverse(view_proj);
            dp.scene_min = sb.mn;
            dp.scene_extent = glm::max(sb.extent(), glm::vec3(1e-4f));
            dp.eye = cam.position();
            dp.sky = cfg.sky;
            display.render(gbuf, solver.target(), refs, dp);
        }

        // 4. UI.
        if (env.nogui) {
            t_imgui.skip();
        } else {
            t_imgui.begin();
            gui.begin_frame();
            ImGui::SetNextWindowSize(ImVec2(430, 700), ImGuiCond_FirstUseEver);
            ImGui::Begin("41 — multi-bounce G-buffer");

            ImGui::Text("%.1f FPS  (%.2f ms)", frame_ms > 0.0 ? 1000.0 / frame_ms : 0.0,
                        t_frame.disp_cpu());

            if (ImGui::CollapsingHeader("Solve", ImGuiTreeNodeFlags_DefaultOpen)) {
                int b = int(cfg.bounces);
                if (ImGui::SliderInt("Bounces (camera levels)", &b, 1, int(kMaxLevels)))
                    cfg.bounces = uint32_t(b);
                int sc = int(cfg.scale);
                if (ImGui::SliderInt("GI scale (1 = per pixel)", &sc, 1, 16))
                    cfg.scale = uint32_t(sc);
                int bu = int(cfg.budget);
                if (ImGui::SliderInt("Budget (cameras/frame)", &bu, 64, 65536))
                    cfg.budget = uint32_t(bu);
                for (uint32_t l = 0; l < cfg.bounces; ++l) {
                    char lbl[32];
                    snprintf(lbl, sizeof lbl, "L%u target", l + 1);
                    int r = int(cfg.res[l]);
                    if (ImGui::SliderInt(lbl, &r, 2, 32)) cfg.res[l] = uint32_t(r);
                    if (l + 1 < cfg.bounces) {
                        snprintf(lbl, sizeof lbl, "L%u spawn tile", l + 1);
                        int k = int(cfg.block[l]);
                        if (ImGui::SliderInt(lbl, &k, 1, 16)) cfg.block[l] = uint32_t(k);
                    }
                }
                ImGui::Checkbox("Running", &cfg.running);
                ImGui::SameLine();
                if (ImGui::Button("Restart sweep")) solver.restart();
                ImGui::SameLine();
                if (ImGui::Button("Clear")) { solver.clear_image(); solver.restart(); }
                ImGui::Text("sweep %u   cursor %u / %u", solver.sweeps(), solver.cursor(),
                            solver.gi_pixels());
                ImGui::Text("cameras this chunk %.0f   texels %.3g",
                            solver.chunk_cameras(), solver.chunk_texels());
                ImGui::Text("camera state %.1f MB", double(solver.bytes()) / (1024.0 * 1024.0));
            }

            if (ImGui::CollapsingHeader("Sky & sun")) {
                // Everything here restarts the sweep: the GI image persists
                // across frames, so a changed environment leaves half of it
                // solved under the old one until the cursor comes round.
                bool touched = false;
                if (ImGui::Button("Daylight")) { cfg.sky = SkyLight::daylight(); touched = true; }
                ImGui::SameLine();
                if (ImGui::Button("Off")) { cfg.sky = SkyLight{}; touched = true; }

                touched |= ImGui::ColorEdit3("Zenith", &cfg.sky.zenith.x,
                                             ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
                touched |= ImGui::ColorEdit3("Horizon", &cfg.sky.horizon.x,
                                             ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
                touched |= ImGui::ColorEdit3("Ground", &cfg.sky.ground.x,
                                             ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
                touched |= ImGui::SliderFloat("Ambient floor", &cfg.sky.ambient.x, 0.0f, 1.0f);
                cfg.sky.ambient.y = cfg.sky.ambient.z = cfg.sky.ambient.x;

                touched |= ImGui::ColorEdit3("Sun irradiance", &cfg.sky.irradiance.x,
                                             ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
                touched |= ImGui::SliderFloat("Sun radius (deg)", &cfg.sky.angle, 0.1f, 10.0f);
                // Azimuth and elevation rather than a raw vector: a direction is
                // the one parameter here that is adjusted by looking at where
                // the shadows land, and dragging three components to keep a unit
                // vector on a sphere is not that.
                {
                    glm::vec3 d = glm::normalize(glm::length(cfg.sky.dir) > 1e-6f
                                                 ? cfg.sky.dir : glm::vec3(0, 1, 0));
                    float elev = glm::degrees(std::asin(std::clamp(d.y, -1.0f, 1.0f)));
                    float azim = glm::degrees(std::atan2(d.x, d.z));
                    bool moved = ImGui::SliderFloat("Sun elevation", &elev, -5.0f, 90.0f);
                    moved |= ImGui::SliderFloat("Sun azimuth", &azim, -180.0f, 180.0f);
                    if (moved) {
                        const float e = glm::radians(elev), a = glm::radians(azim);
                        cfg.sky.dir = glm::vec3(std::cos(e) * std::sin(a), std::sin(e),
                                                std::cos(e) * std::cos(a));
                        touched = true;
                    }
                }
                ImGui::TextUnformatted(
                    cfg.sky.has_sun()
                        ? "sun: analytic magnitude, visibility off a rasterized cone"
                        : "sun off -- its rasterization is skipped entirely");
                if (touched) { solver.clear_image(); solver.restart(); }
            }

            if (ImGui::CollapsingHeader("Transport", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("Emissive scale", &cfg.emissive, 0.0f, 8.0f);
                ImGui::SliderFloat("Camera bias", &cfg.bias, 0.0f,
                                   sb.diagonal() * 0.01f, "%.5f");
                ImGui::SliderFloat("Upsample plane tol", &cfg.plane_tol, 0.0f,
                                   sb.diagonal() * 0.1f, "%.4f");
                ImGui::Checkbox("Force two-sided emitters", &cfg.two_sided);
                ImGui::Checkbox("Analytic direct term", &cfg.nee);
                ImGui::Checkbox("Tent-weighted spawn tiles", &cfg.tent);
                ImGui::Checkbox("Jitter receiver frames", &cfg.jitter);
                {
                    int fi = int(cfg.filter_iters);
                    if (ImGui::SliderInt("GI denoise iters", &fi, 0, 6))
                        cfg.filter_iters = uint32_t(fi);
                    ImGui::SliderInt("GI denoise radius", &cfg.filter_radius, 1, 4);
                }
                ImGui::Checkbox("Indirect only (diagnostic)", &cfg.indirect_only);
                ImGui::Checkbox("Direct term per pixel", &cfg.direct_pixel);
                if (cfg.direct_pixel) {
                    int dr = int(cfg.direct_res);
                    if (ImGui::SliderInt("Light view / pixel", &dr, 2, 32))
                        cfg.direct_res = uint32_t(dr);
                }
                int cr = int(cfg.cam_lv_res);
                if (ImGui::SliderInt("Light view / camera", &cr, 2, 32))
                    cfg.cam_lv_res = uint32_t(cr);
            }

            if (ImGui::CollapsingHeader("View", ImGuiTreeNodeFlags_DefaultOpen)) {
                int n = 0;
                const char* const* names = DisplayPass::view_mode_names(n);
                ImGui::Combo("Mode", &view_mode, names, n);
                ImGui::Combo("Tonemap", &tonemap, kTonemapNames, kTonemapCount);
                ImGui::Combo("Reference", &gt_index, "Direct\0Full GI\0");
                ImGui::SliderFloat("Exposure", &exposure, 0.05f, 8.0f);
                ImGui::SliderFloat("Irradiance gain", &irradiance_gain, 0.05f, 20.0f);
                ImGui::SliderFloat("Diff gain", &diff_gain, 0.5f, 32.0f);
                ImGui::SliderFloat("Split x", &split_x, 0.0f, 1.0f);
                if (ImGui::Button("Reference camera")) {
                    cam.perspective(kGtFovY, cam.aspect(), cam.near_clip(), cam.far_clip());
                    cam.look_at(kGtEye, kGtTarget);
                }
                ImGui::SameLine();
                if (ImGui::Button("Compare now"))
                    compare_to_reference(refs, gt_index, gbuf.width, gbuf.height);
            }

            if (ImGui::CollapsingHeader("Levels", ImGuiTreeNodeFlags_DefaultOpen)) {
                for (uint32_t l = 0; l < solver.levels(); ++l) {
                    const LevelInfo& li = solver.level(l);
                    ImGui::Text("L%u  %2ux%-2u  tile %u  x%-4u  %8u cameras  %6.3f ms",
                                l + 1, li.res, li.res, li.block, li.children, li.cameras,
                                solver.t_raster(l).disp_gpu());
                }
                ImGui::Text("%u triangles, all of them, per camera", scene.count());
            }

            if (ImGui::CollapsingHeader("Timing", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("Frame    %6.2f ms", t_frame.disp_cpu());
                for (PassTimer* t : timers) {
                    if (!t->gpu()) continue;
                    ImGui::Text("%-8s %6.3f ms", t->name(), t->disp_gpu());
                }
                for (PassTimer* t : solver_timers)
                    ImGui::Text("%-8s %6.3f ms", t->name(), t->disp_gpu());
            }

            ImGui::End();
            gui.render();
            t_imgui.end();
        }

        // BEFORE the swap. glfwSwapBuffers leaves the back buffer's contents
        // undefined, so reading it afterwards returns whatever the driver left
        // there -- i.e. a screenshot, or a comparison, that silently lies.
        const bool last_frame =
            env.bench > 0 ? frame_index + 1 >= env.bench : window.should_close();
        if (env.compare && !compare_done && (last_frame || solver.sweeps() > 0)) {
            compare_to_reference(refs, gt_index, gbuf.width, gbuf.height);
            compare_done = true;
        }
        if (env.shot && !shot_done && last_frame) {
            shot_done = gfx::screenshot(env.shot_path.c_str());
            gllib::logf(shot_done ? gllib::LogLevel::info : gllib::LogLevel::error,
                        "%s %s", shot_done ? "wrote" : "FAILED to write",
                        env.shot_path.c_str());
        }

        window.swap_buffers();

        for (PassTimer* t : timers) t->readback();
        for (PassTimer* t : solver_timers) t->readback();
        // Startup frames (shader compilation, the driver's clock ramp) are wildly
        // slower than steady state and would otherwise dominate the first
        // displayed average, which is the only one a short run shows.
        if (frame_index < 30) {
            for (PassTimer* t : timers) t->flush_window();
            for (PassTimer* t : solver_timers) t->flush_window();
            window_accum = 0.0;
        }
        window_accum += frame_ms;
        if (window_accum >= 500.0) {
            for (PassTimer* t : timers) t->flush_window();
            for (PassTimer* t : solver_timers) t->flush_window();
            window_accum = 0.0;
        }

        ++frame_index;

        if (env.bench > 0) {
            bench_ms.push_back(frame_ms);
            if (frame_index >= env.bench) break;
        }
    }

    if (env.shot && !shot_done)
        gllib::logf(gllib::LogLevel::warn, "no screenshot written to %s",
                    env.shot_path.c_str());
    if (env.bench > 0) {
        report_bench(std::move(bench_ms));
        printf("[bench] GPU per pass (avg ms):\n");
        double total = 0.0;
        for (PassTimer* t : timers) {
            if (!t->gpu()) continue;
            printf("    %-16s %7.3f\n", t->name(), t->avg_gpu());
            total += t->avg_gpu();
        }
        for (PassTimer* t : solver_timers) {
            printf("    %-16s %7.3f\n", t->name(), t->avg_gpu());
            total += t->avg_gpu();
        }
        printf("    %-16s %7.3f\n", "GPU TOTAL", total);
        printf("[bench] %u bounces, GI %dx%d, budget %u, targets",
               solver.levels(), solver.gi_size().x, solver.gi_size().y,
               solver.level(0).cameras);
        for (uint32_t l = 0; l < solver.levels(); ++l)
            printf(" %ux%u", solver.level(l).res, solver.level(l).res);
        printf(", %.0f cameras and %.3g texels per chunk\n",
               solver.chunk_cameras(), solver.chunk_texels());
    }

    if (!env.nogui) gui.shutdown();
    return 0;
}

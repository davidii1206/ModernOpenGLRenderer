// ---------------------------------------------------------------------------
// Example 40 — surfel GI, brute-force reference
//
// surfel-gi-spec-r3.md section 10 is emphatic that this method must NOT be
// built in dependency order:
//
//   1. Brute-force all-pairs, no occlusion, no hierarchy. O(N^2), slow, but
//      unambiguously correct for form factors. Do not proceed until this
//      matches.
//   2. Add the microbuffer with a brute-force candidate list (still all-pairs).
//      Now you have correct occlusion. This is the algorithmic core; everything
//      after is acceleration.
//
// This example is exactly those two steps plus the instruments that prove them.
// There is no sparse grid, no FMM, no interaction list and no temporal
// amortization here on purpose: steps 3-10 are acceleration, and step 4 is
// specified as "bit-identical against step 3", which is impossible without this
// reference existing first.
//
// Two path-traced images at a matched camera are the external gate:
//   CornellBoxGroundTruthDirectLighting.png   sweep 1  (direct only)
//   CornellBoxOriginalGroundTruth.png         converged multi-bounce
//
// Design spec: surfel-gi-spec-r3.md in the repo root.
// Stage status: implementation.md next to this file.
//
// Run from this target's build directory: the model, shaders/ and both
// reference PNGs are resolved relative to the working directory.
//
// Every knob is an env var as well as an ImGui control, so a measurement can be
// scripted. The ones that matter, with defaults:
//
//   SGI_GATE=all            run the 30 assertions and exit
//   SGI_MODEL=path.glb      CornellBoxOriginal.glb; the references only match it
//   SGI_SURFELS=30000       target count; the bake floors at one per triangle
//   SGI_BUCKETS=16          microbuffer edge, 8 or 16
//   SGI_BOUNCES=3           max sweeps; sweep k == k bounces under per-pixel NEE
//   SGI_SOLVE=n             pre-solve n sweeps, then hold (scripted shots)
//   SGI_PAUSE=1             hold the solver from frame 0; with SGI_SOLVE=0 the
//                           cache stays zero, which is how a direct-only shot
//                           is taken -- SGI_SOLVE=0 alone keeps solving
//   SGI_METHOD=0|1          M1 analytic (no occlusion) | M2 microbuffer
//   SGI_NEE=1               split the direct term out of the microbuffer
//   SGI_NEE_PIXEL=0         evaluate that direct term per pixel, not per surfel
//   SGI_SKY=0               zenith radiance of an uncovered bucket
//   SGI_SKY_GROUND=         below-horizon radiance; default 0.25 * SGI_SKY
//   SGI_SUN=0               sun radiance, added inside its disc
//   SGI_SUN_ELEV=50         sun elevation in degrees
//   SGI_SUN_AZIM=30         sun azimuth in degrees
//   SGI_SUN_ANGLE=4         sun angular RADIUS in degrees
//   SGI_SUN_NEE=1           NEE owns the sun (sharp shadow); 0 leaves it to the
//                           microbuffer's environment, at 16x16 bucket resolution
//   SGI_NEAR=0              occlusion horizon in SPACINGS; beyond it a surfel
//                           lights but does not block. 0 = unlimited. Simulates
//                           the FMM's U-list horizon (finding 44). Non-zero also
//                           switches the solve to the U-list walk (finding 47)
//   SGI_FAR_OCC=1           march the macro bitmask for the far field's radiance
//   SGI_NEE_SKIP=0          skip the march where the cache's neighbours agree
//                           the light is wholly visible or wholly blocked
//   SGI_NEE_OCC=2.0         occluder radius scale, visibility only (finding 29)
//   SGI_NEE_CUTS=1          clip occluder discs at mesh feature edges
//   SGI_NEE_THICK=0.25      surfel slab half-thickness, in radii
//   SGI_GTCAM=0|1|2         free | reference camera at 512^2 | at 1600x900
//   SGI_BENCH=n             n frames, print per-pass timings, exit
//   SGI_SHOT=path.png       write the final frame
//   SGI_NOGUI=1             no ImGui -- required for a clean screenshot
// ---------------------------------------------------------------------------

#include "gpu_util.hpp"
#include "surfels.hpp"
#include "brute.hpp"
#include "screen.hpp"
#include "grid.hpp"
#include "direct.hpp"
#include "cuts.hpp"
#include "validate.hpp"

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>
#include <gllib/log.hpp>
#include <imgui.h>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace sgi;

namespace {

constexpr const char* kModelPath = "CornellBoxOriginal.glb";

// Camera matched to the two path-traced references.
//
// The glb carries no camera, so this was derived from the reference itself: at
// row 256 of the 512-wide direct-lighting render the back wall spans x in
// [112, 404], i.e. 57.3% of the frame, and the walls are 2.02 units apart, so
// the frame is 3.53 units wide at z = -1.04. That fixes the eye distance for a
// given FOV, and the FOV was then refined against the reference's floor and
// ceiling lines. The scene is NOT fit-normalized (example 39 normalizes because
// it must also handle Sponza), so these are native glb units: floor y = 0,
// ceiling y = 1.99, back wall z = -1.04, opening toward +z.
constexpr glm::vec3 kGtEye{0.004f, 0.999f, 3.864f};
constexpr glm::vec3 kGtTarget{0.004f, 0.999f, 0.0f};
constexpr float     kGtFovY = 38.75f;
constexpr int       kGtRes  = 512;   // square, so the diff is pixel-aligned

const char* const kTonemapNames[] = {"ACES", "Reinhard", "Clamp", "Filmic"};

struct EnvOpts {
    std::string gate;            // SGI_GATE=all|p2p,disc,...   run gates and exit
    std::string model = kModelPath;  // SGI_MODEL=path.glb  (the references only
                                     // match CornellBoxOriginal.glb; anything else
                                     // is for testing the estimator, not for a
                                     // numeric comparison)
    uint32_t surfels = 30000;
    // SGI_SURFELS
    // Two bakes of the same scene at different seeds carry the same lighting, so
    // whatever differs between them is sampling noise and nothing else. That is
    // the only instrument here that can tell noise from real geometric detail,
    // which every high-pass metric tried in this example has failed to do.
    uint32_t seed = 12345u;      // SGI_SEED
    uint32_t budget  = 2048;     // SGI_BUDGET   receivers per frame
    uint32_t bounces = 3;        // SGI_BOUNCES  sweeps; 1 == direct only
    uint32_t buckets = 16;       // SGI_BUCKETS  microbuffer edge (8 or 16)
    int   method = 1;            // SGI_METHOD   0 = M1 radiance, 1 = M2 micro
    float sky = 0.0f;            // SGI_SKY        zenith radiance
    float skyground = -1.0f;     // SGI_SKY_GROUND below-horizon radiance, <0 = 0.25 * sky
    float sun = 0.0f;            // SGI_SUN        sun radiance
    float sunelev = 50.0f;       // SGI_SUN_ELEV   degrees above the horizon
    float sunazim = 30.0f;       // SGI_SUN_AZIM   degrees, 0 = +X toward +Z
    float sunangle = 4.0f;       // SGI_SUN_ANGLE  angular RADIUS in degrees
    bool  sunnee = true;         // SGI_SUN_NEE    1 = NEE owns the sun (sharp shadow)
    float emissive = 1.0f;       // SGI_EMISSIVE
    int   view = 7;              // SGI_VIEW     display mode index
    // 1 = reference camera at 512^2 (pixel-aligned against the PNGs)
    // 2 = reference camera at the normal 1600x900 window. Finding 15's metrics
    //     were all taken at 512^2 and the surfel spacing is 3x coarser in pixels
    //     at full resolution, which is a difference the reconstruction is
    //     sensitive to -- so every reconstruction number now gets taken at both.
    int   gtcam = 0;             // SGI_GTCAM
    uint32_t solve = 0;          // SGI_SOLVE=n  n full sweeps before first present
    int   bench = 0;             // SGI_BENCH=n  n frames, print timing, exit
    bool  shot = false;          // SGI_SHOT=path
    std::string shot_path;
    bool  nogui = false;         // SGI_NOGUI=1
    bool  stats = false;         // SGI_STATS=1  stalling per-sweep read-back
    int   tonemap = 0;           // SGI_TONEMAP
    int   jitter = 1;            // SGI_JITTER   0 none, 1 static, 2 per frame
    bool  points = false;        // SGI_POINTS=1
    int   nee = 1;               // SGI_NEE      1 = split direct out of the microbuffer
    int   neepixel = 0;          // SGI_NEE_PIXEL 1 = direct term per pixel, not per surfel
    float neeskip = 0.0f;        // SGI_NEE_SKIP  cache-agreement margin, 0 = off
    float nearspac = 0.0f;       // SGI_NEAR      occlusion horizon in SPACINGS, 0 = unlimited
    bool  farocc = true;         // SGI_FAR_OCC   march the macro bitmask for the far field
    int   farorder = 1;          // SGI_FAR_ORDER SH bands in the far field: 0 or 1
    // Scripted camera, for scenes that have no reference view. "x,y,z".
    std::string eye, at;         // SGI_EYE / SGI_AT
    float neethick = 0.25f;      // SGI_NEE_THICK surfel slab half-thickness, in radii
    float neeself = 0.9f;        // SGI_NEE_SELF  same-surface normal agreement
    float neeselftol = 1.0f;     // SGI_NEE_SELF_TOL same-surface plane tolerance, in radii
    float neebias = 0.05f;       // SGI_NEE_BIAS receiver offset, in radii
    float neeocc = 2.0f;         // SGI_NEE_OCC  occluder radius scale, visibility only
    int   neecuts = 1;           // SGI_NEE_CUTS 1 = clip discs at mesh feature edges
    bool  showlight = false;     // SGI_SHOW_LIGHT=1 render light_vis instead of E
    int   mls = 1;               // SGI_MLS      0 = Shepard, 1 = degree-1 MLS
    float exposure = 1.0f;       // SGI_EXPOSURE
    bool  paused = false;        // SGI_PAUSE=1
    float bias = -1.0f;          // SGI_BIAS     plane bias, in receiver radii
    float soft = -1.0f;          // SGI_SOFT     disc softening eps, in r^2
    float horizon = -1.0f;       // SGI_HORIZON  receiver-side cos floor
    int   twosided = -1;         // SGI_TWOSIDED force all surfels two-sided
    float cell = 1.0f;           // SGI_CELL     grid cell size, in spacings
    // 1.5, not 2.5. With fat insertion a 3x3x3 window reached about 1.5 spacings
    // whatever this said, so 2.5 was never what the gather actually did; now that
    // the window is sized from the radius, asking for 2.5 costs 3.5x the
    // reconstruction (4.97 ms against 1.41) and measures the same -- MAE 12.94
    // against 12.95, flat from 1.0 to 2.5. This is the value it was.
    float gradius = 1.5f;        // SGI_GATHER_R gather radius, in spacings
    float gplane = 1.0f;         // SGI_GATHER_P gather plane tolerance, in spacings
    float gnormal = 0.0f;        // SGI_GATHER_N gather min dot(n_px, n_surfel)
    int   gkernel = 1;           // SGI_GATHER_K 0 = (r-d) cone, 1 = Wendland C2, 2 = Gaussian
    // The visibility edge stop (findings 19-20) is OFF by default. It was built
    // when the microbuffer carried the direct term, where the cache held a hard
    // shadow boundary that the gather would otherwise interpolate across. Under
    // per-pixel NEE the cache holds the INDIRECT term only, which has no such
    // boundary -- so the stop has nothing legitimate left to preserve, and all it
    // does is suppress neighbours wherever light_vis jumps, carving a rough row
    // out of the reconstruction at every shadow edge. Measured at the ceiling
    // junction, row 109 of the reference camera: the row's horizontal sd was
    // 12.21 against the reference's 4.16, and turning the stop off puts it at
    // 4.40 while the mean error improves from -15.3 to -10.6.
    //
    // Still wired up, because `SGI_NEE_PIXEL=0` does put the direct term back in
    // the cache and then the stop is doing its original job.
    float lsigma = 0.0f;         // SGI_LIGHT_SIGMA edge-stop width, 0 = off
    float lgrad = 0.5f;          // SGI_LIGHT_GRAD  trust in the visibility gradient, 0..1
    int   filter = 2;            // SGI_FILTER   cache denoise iterations, 0 = off
    float fradius = 3.0f;        // SGI_FILTER_R denoise radius, in spacings
};

EnvOpts read_env() {
    EnvOpts o;
    auto u32 = [](const char* v, uint32_t& dst) { dst = uint32_t(std::max(0, atoi(v))); };
    if (const char* v = getenv("SGI_GATE"))     o.gate = v;
    if (const char* v = getenv("SGI_MODEL"))    o.model = v;
    if (const char* v = getenv("SGI_SURFELS"))  u32(v, o.surfels);
    if (const char* v = getenv("SGI_SEED"))     u32(v, o.seed);
    if (const char* v = getenv("SGI_BUDGET"))   u32(v, o.budget);
    if (const char* v = getenv("SGI_BOUNCES"))  u32(v, o.bounces);
    if (const char* v = getenv("SGI_BUCKETS"))  u32(v, o.buckets);
    if (const char* v = getenv("SGI_METHOD"))   o.method = atoi(v);
    if (const char* v = getenv("SGI_SKY"))      o.sky = float(atof(v));
    if (const char* v = getenv("SGI_SKY_GROUND")) o.skyground = float(atof(v));
    if (const char* v = getenv("SGI_SUN"))      o.sun = float(atof(v));
    if (const char* v = getenv("SGI_SUN_ELEV")) o.sunelev = float(atof(v));
    if (const char* v = getenv("SGI_SUN_AZIM")) o.sunazim = float(atof(v));
    if (const char* v = getenv("SGI_SUN_ANGLE")) o.sunangle = float(atof(v));
    if (const char* v = getenv("SGI_SUN_NEE")) o.sunnee = atoi(v) != 0;
    if (const char* v = getenv("SGI_EMISSIVE")) o.emissive = float(atof(v));
    if (const char* v = getenv("SGI_VIEW"))     o.view = atoi(v);
    if (const char* v = getenv("SGI_GTCAM"))    o.gtcam = atoi(v);
    if (const char* v = getenv("SGI_SOLVE"))    u32(v, o.solve);
    if (const char* v = getenv("SGI_BENCH"))    o.bench = atoi(v);
    if (const char* v = getenv("SGI_SHOT"))     { o.shot = true; o.shot_path = v; }
    if (const char* v = getenv("SGI_NOGUI"))    o.nogui = atoi(v) != 0;
    if (const char* v = getenv("SGI_STATS"))    o.stats = atoi(v) != 0;
    if (const char* v = getenv("SGI_TONEMAP"))  o.tonemap = atoi(v);
    if (const char* v = getenv("SGI_JITTER"))   o.jitter = atoi(v);
    if (const char* v = getenv("SGI_POINTS"))   o.points = atoi(v) != 0;
    if (const char* v = getenv("SGI_NEE"))       o.nee = atoi(v);
    if (const char* v = getenv("SGI_NEE_PIXEL")) o.neepixel = atoi(v);
    if (const char* v = getenv("SGI_NEE_SKIP")) o.neeskip = float(atof(v));
    if (const char* v = getenv("SGI_NEE_THICK")) o.neethick = float(atof(v));
    if (const char* v = getenv("SGI_NEE_SELF"))  o.neeself = float(atof(v));
    if (const char* v = getenv("SGI_NEE_SELF_TOL")) o.neeselftol = float(atof(v));
    if (const char* v = getenv("SGI_NEE_BIAS"))  o.neebias = float(atof(v));
    if (const char* v = getenv("SGI_NEE_OCC"))   o.neeocc = float(atof(v));
    if (const char* v = getenv("SGI_NEE_CUTS"))  o.neecuts = atoi(v);
    if (const char* v = getenv("SGI_SHOW_LIGHT")) o.showlight = atoi(v) != 0;
    if (const char* v = getenv("SGI_MLS"))       o.mls = atoi(v);
    if (const char* v = getenv("SGI_EXPOSURE")) o.exposure = float(atof(v));
    if (const char* v = getenv("SGI_PAUSE"))    o.paused = atoi(v) != 0;
    if (const char* v = getenv("SGI_BIAS"))     o.bias = float(atof(v));
    if (const char* v = getenv("SGI_NEAR"))     o.nearspac = float(atof(v));
    if (const char* v = getenv("SGI_FAR_OCC"))  o.farocc = atoi(v) != 0;
    if (const char* v = getenv("SGI_FAR_ORDER")) o.farorder = atoi(v);
    if (const char* v = getenv("SGI_EYE"))      o.eye = v;
    if (const char* v = getenv("SGI_AT"))       o.at = v;
    if (const char* v = getenv("SGI_SOFT"))     o.soft = float(atof(v));
    if (const char* v = getenv("SGI_HORIZON"))  o.horizon = float(atof(v));
    if (const char* v = getenv("SGI_TWOSIDED")) o.twosided = atoi(v);
    if (const char* v = getenv("SGI_CELL"))     o.cell = float(atof(v));
    if (const char* v = getenv("SGI_GATHER_R")) o.gradius = float(atof(v));
    if (const char* v = getenv("SGI_GATHER_P")) o.gplane = float(atof(v));
    if (const char* v = getenv("SGI_GATHER_N")) o.gnormal = float(atof(v));
    if (const char* v = getenv("SGI_GATHER_K")) o.gkernel = atoi(v);
    if (const char* v = getenv("SGI_LIGHT_SIGMA")) o.lsigma = float(atof(v));
    if (const char* v = getenv("SGI_LIGHT_GRAD"))  o.lgrad = float(atof(v));
    if (const char* v = getenv("SGI_FILTER"))   o.filter = atoi(v);
    if (const char* v = getenv("SGI_FILTER_R")) o.fradius = float(atof(v));
    return o;
}

// Percentiles over the steady-state tail. The MINIMUM of the tail is the least
// clock-perturbed sample on a mobile GPU, so it is the number to compare across
// runs; the spread says how noisy the machine was.
void report_bench(std::vector<double> ms) {
    if (ms.empty()) return;
    ms.erase(ms.begin(), ms.begin() + ms.size() / 3);
    if (ms.empty()) return;
    std::sort(ms.begin(), ms.end());
    auto pct = [&](double p) { return ms[std::size_t(p * double(ms.size() - 1))]; };
    printf("[bench] n=%zu  min=%.3f  p10=%.3f  p50=%.3f  p90=%.3f ms\n",
           ms.size(), ms.front(), pct(0.10), pct(0.50), pct(0.90));
}

} // namespace

int main() {
    gllib::log_to_stderr(gllib::LogLevel::info);
    const EnvOpts env = read_env();

    gfx::WindowDesc wd;
    wd.title = "40 — surfel GI, brute-force reference";
    // The reference PNGs are 512x512. Matching the framebuffer exactly makes the
    // split and diff views pixel-aligned against them, which is the whole point
    // of having them.
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

    std::unique_ptr<gfx::Model> model = std::make_unique<gfx::Model>();
    if (!model->load(env.model.c_str())) {
        gllib::logf(gllib::LogLevel::error, "failed to load '%s'", env.model.c_str());
        return 1;
    }

    std::vector<Tri> tris = extract_triangles(*model);
    if (tris.empty()) {
        gllib::log(gllib::LogLevel::error, "no triangles extracted");
        return 1;
    }

    SurfelSet scene;
    scene.build(tris, env.surfels, env.seed);
    apply_base_color_textures(scene, tris, *model);
    const Bounds& sb = scene.bounds();
    gllib::logf(gllib::LogLevel::info,
                "scene: %zu tris, area %.4f, bounds [%.2f %.2f %.2f]..[%.2f %.2f %.2f]",
                tris.size(), total_area(tris), sb.mn.x, sb.mn.y, sb.mn.z,
                sb.mx.x, sb.mx.y, sb.mx.z);
    gllib::logf(gllib::LogLevel::info,
                "surfels: %u (%u emissive), r = %.5f, spacing = %.5f, "
                "sum(pi r^2)/A = %.6f, bake %.2f ms, %.2f MB",
                scene.count(), scene.emissive_count(), scene.radius(), scene.spacing(),
                double(scene.count()) * 3.14159265358979 * double(scene.radius()) *
                    double(scene.radius()) / std::max(1e-12, scene.area()),
                scene.bake_seconds() * 1000.0, double(scene.bytes()) / (1024.0 * 1024.0));

    // --- Passes --------------------------------------------------------------

    GBuffer gbuf;
    gbuf.create(window.framebuffer_width(), window.framebuffer_height());

    GeometryPass geometry;
    SurfelFilterPass filter;
    SurfelGatherPass gather;
    DirectPixelPass direct_px;
    SurfelPointsPass points;
    DisplayPass display;
    Solver solver;
    if (!geometry.init() || !filter.init() || !gather.init() ||
        !direct_px.init() || !points.init() || !display.init() ||
        !solver.init()) {
        gllib::log(gllib::LogLevel::error, "shader initialisation failed");
        return 1;
    }
    gather.resize(gbuf.width, gbuf.height);

    // Static set, so the grid is built once. It is Tier 1's candidate lookup and
    // Tier 2's ray-traversal structure both.
    SurfelGrid grid;
    // Emitter proxies for next event estimation: the direct term is split out
    // of the microbuffer, which resolves this panel with about 4 of 256 buckets.
    EmitterSet emitters;
    // Cut planes: the mesh's own sharp and boundary edges, resolved per surfel,
    // so an occluder disc ends where its geometry does. This is what lets
    // nee_occ inflate the interior without dilating every silhouette -- see
    // cuts.hpp.
    CutSet cuts;

    // Everything downstream of the mesh, in one place so the GUI can load a
    // different one. Every knob that is expressed in SPACINGS has to be
    // recomputed after this, because the spacing changes with the scene -- which
    // is why they are held in their own units and converted per frame rather
    // than baked into cfg once at startup.
    std::string current_model = env.model;
    // Mutable, because the right density is a property of the SCENE and the
    // viewing distance, not a constant. Sponza at the Cornell default of 30000
    // floors at one surfel per triangle -- 285594, spacing 0.168 -- and from
    // inside the arcade one surfel covers about eighty pixels, so the
    // reconstruction draws their Voronoi cells instead of a lit room. At
    // 1387840, spacing 0.076, it resolves.
    uint32_t target_surfels = env.surfels;
    auto rebuild_scene = [&](const char* path) {
        auto next = std::make_unique<gfx::Model>();
        if (!next->load(path)) {
            gllib::logf(gllib::LogLevel::error, "failed to load '%s'", path);
            return false;
        }
        std::vector<Tri> next_tris = extract_triangles(*next);
        if (next_tris.empty()) {
            gllib::logf(gllib::LogLevel::error, "no triangles in '%s'", path);
            return false;
        }
        model = std::move(next);
        tris  = std::move(next_tris);
        scene.build(tris, target_surfels, env.seed);
        apply_base_color_textures(scene, tris, *model);
        grid.build(scene, env.cell);
        emitters.build(tris);
        cuts.build(scene, tris);
        solver.attach(&grid, &emitters, &cuts);
        solver.reset(scene);
        current_model = path;
        const Bounds& b = scene.bounds();
        gllib::logf(gllib::LogLevel::info,
                    "loaded '%s': %zu tris, %u surfels (%u emissive), spacing %.5f, "
                    "bounds [%.2f %.2f %.2f]..[%.2f %.2f %.2f]",
                    path, tris.size(), scene.count(), scene.emissive_count(),
                    scene.spacing(), b.mn.x, b.mn.y, b.mn.z, b.mx.x, b.mx.y, b.mx.z);
        return true;
    };

    grid.build(scene, env.cell);
    emitters.build(tris);
    cuts.build(scene, tris);
    solver.attach(&grid, &emitters, &cuts);

    // Every .glb next to the binary and in the repo's data directory, so the GUI
    // can switch scenes without an env var. Sponza is the reason: it is the only
    // thing here that is not a shoebox, and it cannot be reached any other way
    // from a default launch.
    std::vector<std::string> model_files;
    for (const char* dir : {".", "../../../data"}) {
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
            if (!e.is_regular_file(ec)) continue;
            const std::string ext = e.path().extension().string();
            if (ext == ".glb" || ext == ".gltf") model_files.push_back(e.path().string());
        }
    }
    std::sort(model_files.begin(), model_files.end());

    // Held in their own units and converted every frame: the sun's direction is
    // spherical, and the near horizon is in SPACINGS, which changes with the
    // scene. Baking either into cfg once would break on a model reload.
    float sun_elev = env.sunelev, sun_azim = env.sunazim, sun_angle = env.sunangle;
    float near_spacings = env.nearspac;

    float gather_radius = env.gradius;
    float gather_plane  = env.gplane;
    float gather_normal = env.gnormal;
    int   gather_kernel = env.gkernel;
    float light_sigma   = env.lsigma;
    float grad_scale    = env.lgrad;
    bool  show_light    = env.showlight;
    int   mls_order     = env.mls;
    bool  gather_debug  = getenv("SGI_FALLBACK") != nullptr;
    int   filter_iters  = std::max(0, env.filter);
    float filter_radius = env.fradius;

    References refs;
    refs.load();

    // --- Solver config -------------------------------------------------------

    SolveConfig cfg;
    cfg.method = env.method == 0 ? Method::Radiance : Method::Micro;
    cfg.budget = std::max(1u, env.budget);
    cfg.max_sweeps = env.bounces;
    cfg.ms = env.buckets >= 16 ? 16u : 8u;
    cfg.sky = glm::vec3(env.sky);
    // A ground bounce that is a quarter of the zenith by default: without it an
    // outdoor scene's downward-facing surfaces get nothing at all and read as
    // holes rather than as shadow.
    cfg.sky_ground = glm::vec3(env.skyground >= 0.0f ? env.skyground : 0.25f * env.sky);
    cfg.sun = glm::vec3(env.sun);
    cfg.sun_nee = env.sunnee;
    {
        const float el = glm::radians(env.sunelev), az = glm::radians(env.sunazim);
        cfg.sun_dir = glm::normalize(glm::vec3(std::cos(el) * std::cos(az),
                                               std::sin(el),
                                               std::cos(el) * std::sin(az)));
        cfg.sun_cos = std::cos(glm::radians(std::max(env.sunangle, 0.05f)));
    }
    cfg.emissive_scale = env.emissive;
    cfg.rotate = std::clamp(env.jitter, 0, 2);
    cfg.nee = env.nee != 0;
    cfg.nee_pixel = env.neepixel != 0;
    cfg.nee_thick = env.neethick;
    cfg.nee_self_cos = env.neeself;
    cfg.nee_self_tol = env.neeselftol;
    cfg.nee_bias = env.neebias;
    cfg.nee_occ = env.neeocc;
    cfg.nee_skip = env.neeskip;
    cfg.near_radius = env.nearspac > 0.0f ? env.nearspac * scene.spacing() : 0.0f;
    cfg.far_occlusion = env.farocc;
    cfg.far_order = env.farorder;
    cfg.nee_cuts = env.neecuts != 0;
    if (env.bias >= 0.0f) cfg.plane_bias = env.bias;
    if (env.soft >= 0.0f) cfg.soft_eps = env.soft;
    if (env.horizon >= 0.0f) cfg.horizon = env.horizon;
    if (env.twosided >= 0) cfg.two_sided = env.twosided != 0;
    cfg.running = !env.paused;
    solver.reset(scene);

    {
        double sum_dw = 0.0, sum_wcos = 0.0;
        solver.bucket_sums(cfg.ms, sum_dw, sum_wcos);
        gllib::logf(gllib::LogLevel::info,
                    "bucket table %ux%u: sum(dw) = %.7f (2pi = %.7f), "
                    "sum(wcos) = %.7f (pi = %.7f)",
                    cfg.ms, cfg.ms, sum_dw, 2.0 * 3.14159265358979,
                    sum_wcos, 3.14159265358979);
    }

    // --- Gates ---------------------------------------------------------------
    //
    // Section 9 step 0. They need a GL context but not a frame, so they run here
    // and the process exits: a PI error is invisible in an image and consistent
    // across near and far, so it can only be caught against an analytic answer.
    if (!env.gate.empty()) {
        const bool ok = run_gates(env.gate, scene, solver, cfg, tris);
        if (!env.nogui) gui.shutdown();
        return ok ? 0 : 1;
    }

    // --- Camera --------------------------------------------------------------

    // A reference comparison is only a comparison if the pixel grids match. The
    // window is created at kGtRes but a scaled desktop hands back a bigger
    // framebuffer, so ask for the exact size and say so loudly if it is refused
    // -- silently rendering 640x640 against a 512x512 reference is the kind of
    // measurement that wastes a day.
    if (env.gtcam != 0) {
        const int want_w = env.gtcam == 1 ? kGtRes : 1600;
        const int want_h = env.gtcam == 1 ? kGtRes : 900;
        if (!window.set_framebuffer_size(want_w, want_h))
            gllib::logf(gllib::LogLevel::error,
                        "SGI_GTCAM wanted a %dx%d framebuffer, got %dx%d -- "
                        "reference comparisons at this size are meaningless",
                        want_w, want_h, window.framebuffer_width(),
                        window.framebuffer_height());
    }

    gfx::Camera cam;
    const float radius = std::max(0.1f, sb.radius());
    cam.perspective(env.gtcam != 0 ? kGtFovY : 45.0f,
                    float(window.framebuffer_width()) /
                        float(std::max(1, window.framebuffer_height())),
                    radius * 0.002f, radius * 20.0f);
    auto parse3 = [](const std::string& t, glm::vec3& out) {
        return !t.empty() && std::sscanf(t.c_str(), "%f,%f,%f", &out.x, &out.y, &out.z) == 3;
    };
    glm::vec3 eye_v, at_v;
    if (parse3(env.eye, eye_v) && parse3(env.at, at_v)) {
        // An explicit camera beats both presets: a scene with no reference view
        // has nowhere sensible to put one, and Sponza's default framing is the
        // outside of the building.
        cam.look_at(eye_v, at_v);
    } else if (env.gtcam != 0) {
        cam.look_at(kGtEye, kGtTarget);
    } else {
        cam.look_at(sb.center() + glm::vec3(0.0f, 0.0f, radius * 2.2f), sb.center());
    }

    // --- Timers --------------------------------------------------------------

    PassTimer t_frame("Frame", false);
    PassTimer t_gbuf("G-buffer");
    PassTimer t_recon("Reconstruct");
    PassTimer t_direct_px("Direct/px");
    PassTimer t_display("Display");
    PassTimer t_points("Points");
    PassTimer t_imgui("ImGui");
    PassTimer* const timers[] = {&t_frame, &t_gbuf, &t_recon, &t_direct_px,
                                 &t_display, &t_points, &t_imgui};

    // --- State ---------------------------------------------------------------

    int view_mode = env.view;
    int gt_index = 0;
    int tonemap = std::clamp(env.tonemap, 0, 3);
    float exposure = env.exposure;
    float irradiance_gain = 1.0f;
    float diff_gain = 4.0f;
    float split_x = 0.5f;
    int   point_color = 4;              // irradiance
    float point_scale = 1.0f;
    bool  show_points = env.points;
    bool  collect_stats = env.stats;
    bool  captured = false;
    int   frame_index = 0;
    bool  shot_done = false;
    double last_time = window.time();
    double window_accum = 0.0;
    std::vector<double> bench_ms;

    // Scripted screenshots want a finished solve, not a progressive one -- and
    // then they want it to STOP, so that however many frames the shot takes, the
    // image is exactly n sweeps and not n plus whatever the budget got through.
    if (env.solve > 0) {
        solver.run_sweeps(scene, cfg, env.solve, 0);
        cfg.running = false;
        gllib::logf(gllib::LogLevel::info, "pre-solved %u sweep(s), holding", env.solve);
        // What actually landed in the cache. A scripted shot that comes out black
        // has two very different causes -- an empty cache, or a reconstruction
        // that cannot find it -- and they look identical from the outside.
        {
            const std::vector<glm::vec4> E = scene.read_irradiance();
            double sum = 0.0; uint32_t nz = 0;
            for (const glm::vec4& e : E) {
                const double l = 0.2126 * e.r + 0.7152 * e.g + 0.0722 * e.b;
                sum += l; nz += l > 1e-6 ? 1u : 0u;
            }
            gllib::logf(gllib::LogLevel::info,
                        "cache: mean irradiance %.5f over %zu surfels, %u nonzero (%.1f%%)",
                        E.empty() ? 0.0 : sum / double(E.size()), E.size(), nz,
                        E.empty() ? 0.0 : 100.0 * double(nz) / double(E.size()));
        }
    }

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
            gather.resize(fw, fh);
            cam.set_aspect(float(fw) / float(fh));
        }

        // A scripted measurement must not be steerable. SGI_GTCAM exists to put
        // the camera exactly where the path-traced references were rendered
        // from, and camera_control ran anyway -- so a stray mouse movement while
        // the window came up silently rendered a different view, and every
        // number taken from that shot was wrong while looking perfectly
        // plausible. This example has been bitten by measurements that lie more
        // than by bugs. Use SGI_GTCAM=0 to fly the camera.
        if (env.gtcam == 0)
            camera_control(window, cam, dt, !env.nogui && !gui.wants_mouse(), captured);

        geometry.poll();
        gather.poll();
        filter.poll();
        direct_px.poll();
        points.poll();
        display.poll();
        if (solver.poll()) solver.reset(scene);

        // Derived per frame so the GUI's sliders and a model reload both land.
        {
            const float el = glm::radians(sun_elev), az = glm::radians(sun_azim);
            cfg.sun_dir = glm::normalize(glm::vec3(std::cos(el) * std::cos(az),
                                                   std::sin(el),
                                                   std::cos(el) * std::sin(az)));
            cfg.sun_cos = std::cos(glm::radians(std::max(sun_angle, 0.05f)));
            // NEE's stand-in rectangle: far enough to be outside the scene, and
            // sized to the sun's own solid angle. A disc of angular radius t
            // subtends pi*t^2 and a square of half-extent h at distance D
            // subtends (2h)^2/D^2, so h = D*t*sqrt(pi)/2. Get this wrong and the
            // sun changes brightness rather than shape, which reads as an
            // exposure bug rather than a geometry one.
            const float theta = glm::radians(std::max(sun_angle, 0.05f));
            cfg.sun_dist = std::max(0.1f, 4.0f * scene.bounds().radius());
            cfg.sun_half = cfg.sun_dist * theta * 0.8862269f;
            cfg.near_radius = near_spacings > 0.0f ? near_spacings * scene.spacing() : 0.0f;
        }

        const glm::mat4 view_proj = cam.view_projection();

        // 1. G-buffer.
        {
            ScopedPass p(t_gbuf);
            geometry.render(gbuf, *model, view_proj);
        }

        // 2. One slice of the O(N^2) solve.
        solver.step(scene, cfg, uint32_t(frame_index), collect_stats);

        // 3. Surfel irradiance -> screen, by scatter or by gather.
        {
            ScopedPass p(t_recon);
            // Denoise the cache first, in object space, on a copy. The solve is
            // still iterating on the unfiltered buffer.
            // Version = (sweeps, cursor): the irradiance only changes when the
            // solve advances, so a converged or paused solve reuses the filtered
            // buffer instead of rebuilding it every frame.
            // Bind explicitly rather than relying on the solve having done it:
            // a paused or converged solve does not dispatch, and a stale binding
            // here would silently feed the edge stop garbage.
            if (solver.light_vis_valid()) solver.bind_light_vis();
            const uint64_t irr_version =
                (uint64_t(solver.sweeps()) << 32) | uint64_t(solver.cursor());
            gl::Buffer* filtered =
                filter.run(scene, grid, filter_iters, filter_radius,
                           gather_plane, gather_normal, light_sigma, grad_scale,
                           irr_version);
            gather.render(gbuf, scene, grid, cam, gather_radius, gather_plane,
                          gather_normal, gather_kernel, light_sigma,
                          grad_scale, show_light, mls_order, gather_debug, filtered);
        }

        // 3b. The direct term, per pixel, composited onto whatever the
        //     reconstruction produced. The cache is carrying the residual alone
        //     when this is on (bf_micro.comp's u_nee_add), so this is an add and
        //     not a replacement.
        //
        //     Its own timer, and outside the pass above: PassTimer wraps a GL
        //     query object, and a query cannot be begun while another is active.
        if (cfg.nee && cfg.nee_pixel &&
            (emitters.count() > 0 || cfg.sun_is_nee_light())) {
            ScopedPass p(t_direct_px);
            direct_px.render(gbuf, scene, grid, emitters, cuts, cam,
                             gather.target(), gather.bracket(),
                             gbuf.width, gbuf.height, cfg, show_light);
        } else {
            t_direct_px.skip();
        }

        // 4. Display.
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
            display.render(gbuf, gather.target(), refs, dp);
        }

        // 5. Optional raw point cloud on top — the check that the
        //    reconstruction is not hiding something.
        if (show_points) {
            ScopedPass p(t_points);
            points.render(gbuf, scene, view_proj, cam, point_color, point_scale,
                          irradiance_gain);
        } else {
            t_points.skip();
        }

        // 6. UI.
        if (env.nogui) {
            t_imgui.skip();
        } else {
            t_imgui.begin();
            gui.begin_frame();
            ImGui::SetNextWindowSize(ImVec2(420, 720), ImGuiCond_FirstUseEver);
            ImGui::Begin("40 — brute-force surfel GI");

            ImGui::Text("%.1f FPS  (%.2f ms)", frame_ms > 0.0 ? 1000.0 / frame_ms : 0.0,
                        t_frame.disp_cpu());

            if (ImGui::CollapsingHeader("Solve", ImGuiTreeNodeFlags_DefaultOpen)) {
                int m = int(cfg.method);
                if (ImGui::Combo("Method", &m, "M1 radiance (no occlusion)\0M2 microbuffer\0")) {
                    cfg.method = Method(m);
                    solver.reset(scene);
                }
                int msi = cfg.ms == 8 ? 0 : 1;
                if (ImGui::Combo("Buckets", &msi, "8x8 = 64\0" "16x16 = 256\0")) {
                    cfg.ms = msi == 0 ? 8u : 16u;
                    solver.reset(scene);
                }
                ImGui::Checkbox("Running", &cfg.running);
                ImGui::SameLine();
                if (ImGui::Button("Solve now")) {
                    const uint32_t n = cfg.max_sweeps ? cfg.max_sweeps : 8u;
                    solver.reset(scene);
                    solver.run_sweeps(scene, cfg, n, uint32_t(frame_index));
                }
                ImGui::SameLine();
                if (ImGui::Button("Reset")) solver.reset(scene);

                int budget = int(cfg.budget);
                if (ImGui::SliderInt("Budget / frame", &budget, 64,
                                     int(std::max(64u, scene.count()))))
                    cfg.budget = uint32_t(std::max(1, budget));
                int sweeps = int(cfg.max_sweeps);
                if (ImGui::SliderInt("Max sweeps (bounces)", &sweeps, 1, 64))
                    cfg.max_sweeps = uint32_t(std::max(1, sweeps));

                ImGui::Text("sweep %u / %u   cursor %u / %u", solver.sweeps(),
                            cfg.max_sweeps, solver.cursor(), scene.count());
                const SolveStats& st = solver.stats();
                if (st.valid) {
                    ImGui::Text("mean E %.5f  peak %.4f", st.mean, st.peak);
                    ImGui::Text("flux %.5f W  delta %.3e (%.4f%%)",
                                st.flux, st.delta, st.rel_delta * 100.0);
                    ImGui::Text("nonzero %u / %u", st.nonzero, scene.count());
                }
                ImGui::Checkbox("Collect stats (stalls)", &collect_stats);
            }

            if (ImGui::CollapsingHeader("Transport", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("Emissive scale", &cfg.emissive_scale, 0.0f, 8.0f);
                // The environment a bucket sees when it sees no geometry. The
                // sun is a disc in it rather than an emitter proxy, so its
                // shadow comes from the microbuffer's own occlusion -- and is
                // only as sharp as 16x16 buckets, which is why its angular
                // radius wants to be degrees rather than the real 0.53.
                ImGui::SliderFloat("Sky zenith", &cfg.sky.x, 0.0f, 4.0f);
                cfg.sky.y = cfg.sky.z = cfg.sky.x;
                ImGui::SliderFloat("Sky ground", &cfg.sky_ground.x, 0.0f, 4.0f);
                cfg.sky_ground.y = cfg.sky_ground.z = cfg.sky_ground.x;
                ImGui::SliderFloat("Sun", &cfg.sun.x, 0.0f, 40.0f);
                cfg.sun.y = cfg.sun.z = cfg.sun.x;
                ImGui::SliderFloat("Sun elevation", &sun_elev, 0.0f, 90.0f, "%.0f deg");
                ImGui::SliderFloat("Sun azimuth", &sun_azim, -180.0f, 180.0f, "%.0f deg");
                ImGui::SliderFloat("Sun radius", &sun_angle, 0.5f, 20.0f, "%.1f deg");
                // Exactly one of the two must own the sun or it is counted
                // twice. NEE traces it with the same 256-bit cone mask the
                // emitter rectangles get, so its shadow is as sharp as theirs;
                // the microbuffer's is 16x16 buckets wide.
                ImGui::Checkbox("Sun through NEE (sharp shadow)", &cfg.sun_nee);
                // Above 65536 surfels the all-pairs path cannot index its own
                // winners, so a big scene NEEDS this non-zero. Sponza is 285594.
                if (ImGui::SliderFloat("Near horizon", &near_spacings, 0.0f, 8.0f,
                                       "%.2f spacings"))
                    solver.reset(scene);
                if (scene.count() > 65536u && near_spacings <= 0.0f)
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.3f, 1),
                                       "%u surfels: raise the near horizon above 0",
                                       scene.count());
                ImGui::SliderFloat("Horizon cos", &cfg.horizon, 0.0f, 0.2f, "%.4f");
                ImGui::SliderFloat("Plane bias (radii)", &cfg.plane_bias, 0.0f, 4.0f);
                ImGui::SliderFloat("Disc softening eps", &cfg.soft_eps, 0.0f, 4.0f);
                ImGui::SliderFloat("Depth tol (radii)", &cfg.depth_tol_radii, 0.0f, 8.0f);
                ImGui::SliderFloat("Normal tol (cos)", &cfg.normal_tol, -1.0f, 1.0f);
                ImGui::Checkbox("Two-sided emitters", &cfg.two_sided);
                ImGui::SameLine();
                ImGui::Checkbox("No occlusion (gate 5)", &cfg.no_occlusion);
                ImGui::Combo("Frame rotate", &cfg.rotate,
                             "None\0Static per surfel\0Per surfel per frame\0");
            }

            if (ImGui::CollapsingHeader("Direct (NEE)", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Split direct out of the microbuffer", &cfg.nee);
                ImGui::Checkbox("Per pixel (not per surfel)", &cfg.nee_pixel);
                if (cfg.nee_pixel) {
                    ImGui::TextDisabled("sweep k == k bounces here; per-surfel needs k+1");
                }
                ImGui::Checkbox("Clip discs at mesh edges", &cfg.nee_cuts);
                ImGui::SliderFloat("Occluder radius scale", &cfg.nee_occ, 1.0f, 4.0f);
                ImGui::SliderFloat("Slab thickness (radii)", &cfg.nee_thick, 0.0f, 1.0f);
                ImGui::SliderFloat("Receiver bias (radii)", &cfg.nee_bias, 0.0f, 1.0f);
                ImGui::SliderFloat("Same-surface cos", &cfg.nee_self_cos, -1.0f, 1.0f);
                ImGui::SliderFloat("Same-surface tol (radii)", &cfg.nee_self_tol, 0.0f, 4.0f);
            }

            if (ImGui::CollapsingHeader("View", ImGuiTreeNodeFlags_DefaultOpen)) {
                int n = 0;
                const char* const* names = DisplayPass::view_mode_names(n);
                ImGui::Combo("Mode", &view_mode, names, n);
                ImGui::Combo("Tonemap", &tonemap, kTonemapNames, 4);
                ImGui::Combo("Reference", &gt_index, "Direct\0Full GI\0");
                ImGui::SliderFloat("Exposure", &exposure, 0.05f, 8.0f);
                ImGui::SliderFloat("Irradiance gain", &irradiance_gain, 0.05f, 20.0f);
                ImGui::SliderFloat("Diff gain", &diff_gain, 0.5f, 32.0f);
                ImGui::SliderFloat("Split x", &split_x, 0.0f, 1.0f);
                {
                    ImGui::SliderFloat("Light edge stop", &light_sigma, 0.0f, 0.2f,
                                       "%.4f");
                    ImGui::SliderFloat("Visibility gradient", &grad_scale, 0.0f, 1.0f);
                    ImGui::Checkbox("Show light visibility", &show_light);
                    ImGui::SliderInt("Cache denoise iters", &filter_iters, 0, 24);
                    ImGui::SliderFloat("Denoise radius (spacings)", &filter_radius, 0.5f, 6.0f);
                    ImGui::Combo("Interpolation", &mls_order,
                                 "Shepard (degree 0)\0" "Moving least squares (degree 1)\0");
                    ImGui::Combo("Gather kernel", &gather_kernel,
                                 "Schaufler-Jensen (r-d) cone\0" "Wendland C2\0" "Gaussian\0");
                    ImGui::SliderFloat("Gather radius (spacings)", &gather_radius, 0.1f, 4.0f);
                    ImGui::SliderFloat("Gather plane tol (spacings)", &gather_plane, 0.1f, 3.0f);
                    ImGui::SliderFloat("Gather normal tol", &gather_normal, -0.2f, 0.95f);
                    ImGui::Checkbox("Show fallback pixels", &gather_debug);
                    ImGui::Text("grid %dx%dx%d  %u entries  %.1f/occupied cell  max %u",
                                grid.res().x, grid.res().y, grid.res().z, grid.entries(),
                                grid.mean_per_occupied(), grid.max_per_cell());
                }

                ImGui::Checkbox("Surfel points", &show_points);
                if (show_points) {
                    int cn = 0;
                    const char* const* cnames = SurfelPointsPass::color_mode_names(cn);
                    ImGui::Combo("Point colour", &point_color, cnames, cn);
                    ImGui::SliderFloat("Point scale", &point_scale, 0.25f, 8.0f);
                }
                if (ImGui::Button("Reference camera")) {
                    cam.perspective(kGtFovY, cam.aspect(), cam.near_clip(), cam.far_clip());
                    cam.look_at(kGtEye, kGtTarget);
                }
            }

            if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
                // Loading a different mesh rebuilds the surfels, the grid, the
                // emitter proxies and the cut planes, and resets the solve.
                // Sponza is the whole reason this exists: it is 262266 triangles
                // against Cornell's 32, and a default launch has no other way to
                // reach it.
                if (!model_files.empty()) {
                    static int sel = 0;
                    std::vector<const char*> names(model_files.size());
                    for (std::size_t k = 0; k < model_files.size(); ++k)
                        names[k] = model_files[k].c_str();
                    ImGui::SetNextItemWidth(260.0f);
                    ImGui::Combo("##model", &sel, names.data(), int(names.size()));
                    ImGui::SameLine();
                    if (ImGui::Button("Load")) {
                        if (rebuild_scene(model_files[std::size_t(sel)].c_str())) {
                            // Frame the new scene: its bounds have nothing to do
                            // with the old one's, and a camera left where it was
                            // is usually inside a wall or a mile away.
                            const float r = std::max(0.1f, scene.bounds().radius());
                            cam.perspective(45.0f,
                                            float(gbuf.width) / float(std::max(1, gbuf.height)),
                                            r * 0.002f, r * 20.0f);
                            cam.look_at(scene.bounds().center() +
                                            glm::vec3(0.0f, 0.0f, r * 2.2f),
                                        scene.bounds().center());

                            // Make the load land somewhere it can be seen.
                            //
                            // A scene with no emissive surface has no light at
                            // all in this renderer, and one past 65536 surfels
                            // cannot use the all-pairs path -- so loading Sponza
                            // with the Cornell defaults gives a black screen for
                            // two reasons at once, neither of which is obvious
                            // from looking at it.
                            if (scene.count() > 65536u && near_spacings <= 0.0f)
                                near_spacings = 1.5f;
                            if (scene.emissive_count() == 0 &&
                                cfg.sky.x <= 0.0f && cfg.sun.x <= 0.0f) {
                                cfg.sky = glm::vec3(0.6f);
                                cfg.sky_ground = glm::vec3(0.15f);
                                cfg.sun = glm::vec3(8.0f);
                            }
                        }
                    }
                    ImGui::TextUnformatted(current_model.c_str());
                }
                // Density. The bake floors at one surfel per triangle, so this is
                // a target rather than a count, and the spacing it produces is
                // the number that matters: when a surfel covers more than a few
                // pixels the gather has nothing to interpolate between and falls
                // back to the nearest one, which draws Voronoi cells.
                int want = int(target_surfels);
                ImGui::SetNextItemWidth(180.0f);
                if (ImGui::DragInt("Target surfels", &want, 5000.0f, 1000, 4000000))
                    target_surfels = uint32_t(std::max(want, 1000));
                ImGui::SameLine();
                if (ImGui::Button("Rebuild")) rebuild_scene(current_model.c_str());
                ImGui::Text("%u surfels, spacing %.4f, %.0f MB grid",
                            scene.count(), scene.spacing(),
                            double(grid.bytes()) / (1024.0 * 1024.0));
                ImGui::Separator();

                ImGui::Text("surfels     %u (%u emissive)", scene.count(),
                            scene.emissive_count());
                ImGui::Text("radius      %.5f", scene.radius());
                ImGui::Text("spacing     %.5f", scene.spacing());
                ImGui::Text("area        %.4f", scene.area());
                ImGui::Text("pairs/sweep %.3g",
                            double(scene.count()) * double(scene.count()));
                ImGui::Text("bake        %.2f ms", scene.bake_seconds() * 1000.0);
                ImGui::Text("eye   %.3f %.3f %.3f", cam.position().x, cam.position().y,
                            cam.position().z);
                ImGui::Text("target %.3f %.3f %.3f", cam.target().x, cam.target().y,
                            cam.target().z);
                ImGui::Text("fov   %.2f", cam.fov());
            }

            if (ImGui::CollapsingHeader("Timing", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("Frame  %6.2f ms", t_frame.disp_cpu());
                ImGui::Text("Solve  %6.3f ms", solver.timer().disp_gpu());
                for (PassTimer* t : timers) {
                    if (!t->gpu()) continue;
                    ImGui::Text("%-8s %6.3f ms", t->name(), t->disp_gpu());
                }
            }

            ImGui::End();
            gui.render();
            t_imgui.end();
        }

        // BEFORE the swap. glfwSwapBuffers leaves the back buffer's contents
        // undefined, so reading it afterwards returns whatever the driver left
        // there -- which on this machine is intermittently all-black or
        // all-white, i.e. a screenshot that silently lies about the render.
        if (env.shot && !shot_done &&
            (env.bench > 0 ? frame_index + 1 >= env.bench : window.should_close())) {
            shot_done = gfx::screenshot(env.shot_path.c_str());
            gllib::logf(shot_done ? gllib::LogLevel::info : gllib::LogLevel::error,
                        "%s %s", shot_done ? "wrote" : "FAILED to write",
                        env.shot_path.c_str());
        }

        window.swap_buffers();

        for (PassTimer* t : timers) t->readback();
        solver.timer().readback();
        // Startup frames (shader compilation, the bake, the driver's clock ramp)
        // are wildly slower than steady state and would otherwise dominate the
        // first displayed average, which is the only one a short run shows.
        if (frame_index < 30) {
            for (PassTimer* t : timers) t->flush_window();
            solver.timer().flush_window();
            window_accum = 0.0;
        }
        window_accum += frame_ms;
        if (window_accum >= 500.0) {
            for (PassTimer* t : timers) t->flush_window();
            solver.timer().flush_window();
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
        printf("    %-16s %7.3f\n", "Solve", solver.timer().avg_gpu());
        total += solver.timer().avg_gpu();
        for (PassTimer* t : timers) {
            if (!t->gpu()) continue;
            printf("    %-16s %7.3f\n", t->name(), t->avg_gpu());
            total += t->avg_gpu();
        }
        printf("    %-16s %7.3f\n", "GPU TOTAL", total);
        printf("[bench] %u surfels, %s, %ux%u buckets, budget %u\n",
               scene.count(), cfg.method == Method::Micro ? "M2 micro" : "M1 radiance",
               cfg.ms, cfg.ms, cfg.budget);
    }

    if (!env.nogui) gui.shutdown();
    return 0;
}

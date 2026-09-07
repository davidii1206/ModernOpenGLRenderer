// ---------------------------------------------------------------------------
// Example 39 — MOSAIC
// Micro-rendered Object-Space Amortized Irradiance Cache
//
// Indirect light is cached on OBJECT-SPACE surfel sets that ship with the mesh
// asset, so rigid and skinned motion transforms the cache instead of
// invalidating it. Transport is estimated by micro-rasterizing a camera-centred
// clipmap of surfel clusters into tiny per-surfel octahedral buffers rather
// than by tracing rays: no BVH, no acceleration-structure refit, no hardware
// ray tracing. The hemisphere integral is split by frequency — a deterministic
// 16x16 micro-buffer splat captures the low-frequency bulk, while world-space
// ReSTIR reservoirs importance-sample the bright tail the micro-buffer would
// undersample. Direct light on surfels is re-evaluated every frame and only the
// bounce term is amortized, so light response is one frame per bounce rather
// than a convergence window.
//
// Design spec: "Mosaic Lighting.md" in the repo root.
// Stage status:  implementation.md next to this file.
//
// Prior art in this repo that this example builds on:
//   36_micro_gi    Ritschel et al. 2009 micro-rendering via point splatting
//   37_emissive_gi packed surfel records + imageAtomicMin micro-buffer
//   38_surfel_rt   GPU sparse grids, counting sort, cluster merging, ray streams
//   33_mdc_voxelize bit-packed occupancy grids + Amanatides & Woo DDA
//
// Run from the build directory of this target: models, shaders/ and cache/ are
// all resolved relative to the working directory.
// ---------------------------------------------------------------------------

#include "gpu_util.hpp"
#include "mosaic.hpp"
#include "direct.hpp"
#include "screen.hpp"
#include "surfel_bake.hpp"
#include "clipmap.hpp"
#include "cache_gi.hpp"

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>
#include <gllib/log.hpp>
#include <imgui.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace mosaic;

namespace {

// --- Scene definitions ------------------------------------------------------

struct SceneDef {
    const char* name;
    const char* path;
    bool add_movers;          // spawn the rigid dynamic instances
    // Default camera, in units of the scene radius relative to its centre.
    // Sponza needs an interior viewpoint: the default "back off along +Z"
    // placement stares at the outside of the atrium wall.
    glm::vec3 eye_rel;
    glm::vec3 target_rel;
};

const SceneDef kScenes[] = {
    // Just outside the open front of the box, looking in. Cascade 0 is
    // camera-centred, so this standoff is exactly what the C0 coverage floor
    // below has to accommodate.
    {"Cornell Box", "CornellBoxOriginal.glb", true,
     {0.0f, 0.0f, 1.25f}, {0.0f, 0.0f, 0.0f}},
    // Eye pulled forward along the view direction: at 0.62 the camera sat
    // inside the lion relief on the +X end wall.
    {"Sponza",      "sponza.glb",             true,
     {0.30f, -0.22f, 0.0f}, {-0.62f, -0.18f, 0.0f}},
};

// Separate asset for the rigid movers. Reusing a scene mesh gives flat panels
// on Cornell, which is useless for judging whether cached irradiance travels
// with a moving object; a compact closed mesh reads immediately. It also
// exercises the multi-model instance path that S0 needs anyway.
constexpr const char* kMoverModelPath = "Stanford_Bunny.glb";

// LOD-0 surfel budget for the whole scene. The live working set is capped
// separately by Config::kMaxLiveSurfels; this only sets the bake density.
constexpr uint32_t kTargetLod0Surfels = 400000;

// Normalizes any model to a comfortable working scale and centres it on the
// origin. Sponza spans ~2735 units and Cornell ~2, and every distance-based
// heuristic downstream (cascade cell size, surfel spacing, gather radii) is
// easier to reason about against a known extent.
// When `subset` is non-empty only those meshes are considered. That matters for
// LOD-showcase assets such as Stanford_Bunny.glb, whose seven LOD meshes are
// laid out SIDE BY SIDE rather than coincident: fitting to the union of all of
// them shrinks the one mesh actually drawn by the number of levels.
glm::mat4 fit_transform(const gfx::Model& model, float target_extent, Bounds& out_local,
                        const std::vector<size_t>& subset = {}) {
    Bounds b;
    auto accumulate = [&](size_t i) {
        const glm::mat4& xf = model.mesh_transform(i);
        for (const gfx::Vertex& v : model.mesh(i).vertices())
            b.add(glm::vec3(xf * glm::vec4(v.position[0], v.position[1], v.position[2], 1.0f)));
    };
    if (subset.empty()) {
        for (size_t i = 0; i < model.mesh_count(); ++i) accumulate(i);
    } else {
        for (size_t i : subset) if (i < model.mesh_count()) accumulate(i);
    }
    out_local = b;
    if (!b.valid()) return glm::mat4(1.0f);

    const glm::vec3 ext = b.extent();
    const float span = std::max({ext.x, ext.y, ext.z, 1e-6f});
    const float s = target_extent / span;
    return glm::scale(glm::mat4(1.0f), glm::vec3(s)) *
           glm::translate(glm::mat4(1.0f), -b.center());
}

// Environment hooks, mirroring the SRT_* set in example 38 so the two examples
// can be benchmarked and screenshotted the same way from a script.
struct EnvOpts {
    int   scene = 0;
    int   bench = 0;         // MOSAIC_BENCH=N: run N frames, print timing, exit
    int   view = 0;
    bool  points = false;    // MOSAIC_POINTS=1: start with the surfel overlay on
    bool  movers = true;     // MOSAIC_MOVERS=0: static scene only
    bool  validate = false;  // MOSAIC_VALIDATE=1: run the GPU-vs-CPU checks once
    float sh_test = 0.0f;    // MOSAIC_SH_TEST=L: constant-radiance SH round trip
    float emissive = -1.0f;  // MOSAIC_EMISSIVE=f: emissive boost override
    bool  have_eye = false;  // MOSAIC_EYE="x,y,z" in units of the scene radius
    glm::vec3 eye{0.0f};
    bool  have_target = false;   // MOSAIC_TARGET="x,y,z"
    glm::vec3 target{0.0f};
    int   surfel_color = 0;  // MOSAIC_SURFEL_COLOR=n
    int   surfel_lod = -1;   // MOSAIC_SURFEL_LOD=n (-1 = the live LOD)
    float point_scale = 1.0f;// MOSAIC_POINT_SCALE=f
    float irradiance_gain = 1.0f;   // MOSAIC_IRR_GAIN=f
    // The cache carries a bounce per frame through the S2 feedback loop rather
    // than a converged multi-bounce solution, so an indoor scene lands about a
    // stop under a path-traced reference at unit exposure.
    float exposure = 2.0f;   // MOSAIC_EXPOSURE=f
    bool  no_gui = false;    // MOSAIC_NOGUI=1: skip the ImGui overlay
    float reach = 4.0f;      // MOSAIC_REACH=f: screen-gather reach, in surfel radii
    int   shell = 3;         // MOSAIC_SHELL=n: S5 shell radius in cells
    float spatial = -1.0f;   // MOSAIC_SPATIAL=f: S6b spatial filter strength
    float sky = -1.0f;       // MOSAIC_SKY=f: uniform sky radiance override
    uint32_t budget = 0;     // MOSAIC_BUDGET=n: surfel updates per frame
    float tmax = -1.0f;      // MOSAIC_TMAX=f: temporal alpha upper bound
    int   esteps = -1;       // MOSAIC_ESTEPS=n: emitter cone-march steps (0 = unshadowed)
    bool  shot = false;      // MOSAIC_SHOT=path
    std::string shot_path;
};

EnvOpts read_env() {
    EnvOpts o;
    if (const char* v = getenv("MOSAIC_SCENE")) o.scene = atoi(v);
    if (const char* v = getenv("MOSAIC_BENCH")) o.bench = atoi(v);
    if (const char* v = getenv("MOSAIC_VIEW"))  o.view = atoi(v);
    if (const char* v = getenv("MOSAIC_POINTS")) o.points = atoi(v) != 0;
    if (const char* v = getenv("MOSAIC_MOVERS")) o.movers = atoi(v) != 0;
    if (const char* v = getenv("MOSAIC_VALIDATE")) o.validate = atoi(v) != 0;
    if (const char* v = getenv("MOSAIC_SH_TEST")) o.sh_test = float(atof(v));
    if (const char* v = getenv("MOSAIC_EMISSIVE")) o.emissive = float(atof(v));
    auto parse_vec3 = [](const char* v, glm::vec3& out) {
        return sscanf(v, "%f,%f,%f", &out.x, &out.y, &out.z) == 3;
    };
    if (const char* v = getenv("MOSAIC_EYE")) o.have_eye = parse_vec3(v, o.eye);
    if (const char* v = getenv("MOSAIC_TARGET")) o.have_target = parse_vec3(v, o.target);
    if (const char* v = getenv("MOSAIC_SURFEL_COLOR")) o.surfel_color = atoi(v);
    if (const char* v = getenv("MOSAIC_SURFEL_LOD")) o.surfel_lod = atoi(v);
    if (const char* v = getenv("MOSAIC_POINT_SCALE")) o.point_scale = float(atof(v));
    if (const char* v = getenv("MOSAIC_IRR_GAIN")) o.irradiance_gain = float(atof(v));
    if (const char* v = getenv("MOSAIC_EXPOSURE")) o.exposure = float(atof(v));
    if (const char* v = getenv("MOSAIC_NOGUI")) o.no_gui = atoi(v) != 0;
    if (const char* v = getenv("MOSAIC_REACH")) o.reach = float(atof(v));
    if (const char* v = getenv("MOSAIC_SHELL")) o.shell = atoi(v);
    if (const char* v = getenv("MOSAIC_SPATIAL")) o.spatial = float(atof(v));
    if (const char* v = getenv("MOSAIC_SKY")) o.sky = float(atof(v));
    if (const char* v = getenv("MOSAIC_BUDGET")) o.budget = uint32_t(atoi(v));
    if (const char* v = getenv("MOSAIC_TMAX")) o.tmax = float(atof(v));
    if (const char* v = getenv("MOSAIC_ESTEPS")) o.esteps = atoi(v);
    if (const char* v = getenv("MOSAIC_SHOT"))  { o.shot = true; o.shot_path = v; }
    return o;
}

// Percentiles over the steady-state tail of a frame-time series. The MINIMUM
// of the tail is the least clock-perturbed sample on a mobile GPU, so it is
// the number to compare across runs; the spread says how noisy the machine was.
void report_bench(std::vector<double> ms) {
    if (ms.empty()) return;
    // Drop the first third: shader compilation, cache warmup and the driver's
    // clock ramp all live there.
    ms.erase(ms.begin(), ms.begin() + ms.size() / 3);
    std::sort(ms.begin(), ms.end());
    auto pct = [&](double p) { return ms[size_t(p * double(ms.size() - 1))]; };
    printf("[bench] n=%zu  min=%.3f  p10=%.3f  p50=%.3f  p90=%.3f ms\n",
           ms.size(), ms.front(), pct(0.10), pct(0.50), pct(0.90));
}

} // namespace

int main() {
    gllib::log_to_stderr(gllib::LogLevel::info);
    const EnvOpts env = read_env();

    gfx::WindowDesc wd;
    wd.title = "39 — MOSAIC";
    wd.width = 1600;
    wd.height = 900;
    wd.vsync = false;
    wd.debug = true;
    gfx::Window window(wd);
    window.vsync(false);
    gl::enable_debug_output(false);

    gfx::ImGuiOverlay gui;
    gui.init(window);

    // --- Scene ---------------------------------------------------------------
    const int scene_idx = std::clamp(env.scene, 0, int(sizeof(kScenes) / sizeof(kScenes[0])) - 1);
    const SceneDef& scene = kScenes[scene_idx];

    auto model = std::make_unique<gfx::Model>();
    if (!model->load(scene.path)) {
        gllib::logf(gllib::LogLevel::error, "failed to load '%s'", scene.path);
        return 1;
    }

    Bounds local_bounds;
    const glm::mat4 fit = fit_transform(*model, 4.0f, local_bounds);

    std::vector<const gfx::Model*> models{model.get()};
    std::vector<std::string> model_paths{scene.path};
    std::vector<Instance> instances;
    for (size_t i = 0; i < model->mesh_count(); ++i) {
        Instance inst;
        inst.model = 0;
        inst.mesh = int(i);
        inst.material = model->mesh_material(i);
        inst.xform = fit * model->mesh_transform(i);
        inst.prev_xform = inst.xform;
        instances.push_back(inst);
    }
    const size_t static_instance_count = instances.size();

    // World-space scene bounds after fitting, used for cascade sizing and the
    // world-position debug view.
    Bounds scene_bounds;
    for (const Instance& inst : instances) {
        if (size_t(inst.mesh) >= model->mesh_count()) continue;
        for (const gfx::Vertex& v : model->mesh(size_t(inst.mesh)).vertices())
            scene_bounds.add(glm::vec3(inst.xform *
                glm::vec4(v.position[0], v.position[1], v.position[2], 1.0f)));
    }

    // Rigid movers: a few instances that orbit through the scene. They exist to
    // exercise the object-space claim — their cached irradiance must translate
    // with them rather than re-converge.
    // Mover asset, fitted to a fraction of the scene so it sits inside the room.
    auto mover_model = std::make_unique<gfx::Model>();
    std::vector<glm::mat4> mover_base;
    if (scene.add_movers && env.movers && mover_model->load(kMoverModelPath)) {
        models.push_back(mover_model.get());
        model_paths.push_back(kMoverModelPath);

        // Take LOD 0 of each LOD group, or every mesh when the asset has no
        // LOD groups. Stanford_Bunny.glb is one group of 7 levels, so iterating
        // meshes blindly spawns seven bunnies per orbit slot.
        std::vector<size_t> mover_meshes;
        if (mover_model->lod_group_count() > 0) {
            for (size_t g = 0; g < mover_model->lod_group_count(); ++g) {
                const gfx::LodGroup& grp = mover_model->lod_group(g);
                if (!grp.mesh_indices.empty()) mover_meshes.push_back(size_t(grp.mesh_indices[0]));
            }
        } else {
            for (size_t m = 0; m < mover_model->mesh_count(); ++m) mover_meshes.push_back(m);
        }

        Bounds mover_local;
        const glm::mat4 mover_fit = fit_transform(
            *mover_model, scene_bounds.radius() * 0.35f, mover_local, mover_meshes);
        gllib::logf(gllib::LogLevel::info,
                    "mover '%s': %zu mesh(es), extent %.3f x %.3f x %.3f -> %.3f",
                    kMoverModelPath, mover_meshes.size(), mover_local.extent().x,
                    mover_local.extent().y, mover_local.extent().z,
                    scene_bounds.radius() * 0.35f);

        const glm::vec3 c = scene_bounds.center();
        const float r = scene_bounds.radius();
        for (size_t m : mover_meshes) {
            for (int k = 0; k < 3; ++k) {
                Instance mv;
                mv.model = int(models.size()) - 1;
                mv.mesh = int(m);
                mv.material = mover_model->mesh_material(m);
                mv.flags = kInstanceDynamic;
                mv.orbit_center = c + glm::vec3(0.0f, scene_bounds.extent().y * 0.30f, 0.0f);
                mv.orbit_radius = r * 0.42f;
                mv.orbit_speed = 0.35f + 0.15f * float(k);
                mv.orbit_phase = float(k) * 2.0944f;   // 120 degrees apart
                mv.spin_speed = 0.8f;
                instances.push_back(mv);
                mover_base.push_back(mover_fit * mover_model->mesh_transform(m));
            }
        }
    } else if (scene.add_movers && env.movers) {
        gllib::logf(gllib::LogLevel::warn, "mover model '%s' not found; scene is fully static",
                    kMoverModelPath);
    }

    gllib::logf(gllib::LogLevel::info,
                "scene '%s': %zu meshes, %zu instances (%zu movers), extent %.2f x %.2f x %.2f",
                scene.name, model->mesh_count(), instances.size(),
                instances.size() - static_instance_count,
                scene_bounds.extent().x, scene_bounds.extent().y, scene_bounds.extent().z);

    Config cfg;   // surfel_spacing is filled in after the bake below

    // --- M1: surfel bake -----------------------------------------------------
    //
    // LOD-0 spacing comes from a surfel BUDGET over the scene's actual surface
    // area, not from the cascade cell size. The cascade size says nothing about
    // how much surface a scene contains: deriving spacing from it undersamples
    // Cornell and oversamples Sponza by two orders of magnitude. The spec's LOD
    // ratios (4/8/25/100 cm) are then applied on top by bake_mesh.
    const double scene_area = scene_surface_area(instances, models);
    BakeParams bake;
    bake.world_spacing = spacing_for_budget(scene_area, kTargetLod0Surfels);
    cfg.surfel_spacing = bake.world_spacing;
    // Cascade 0's cell follows the SURFEL SPACING, not the scene radius: the
    // spec pairs a 0.5 m cell with 4 cm surfels, i.e. about 12 surfels across a
    // cell. Sizing it from the scene radius instead makes C0 barely wider than
    // the scene, so any framing viewpoint leaves the subject outside the finest
    // cascade entirely.
    cfg.cascade0_cell = std::max(cfg.surfel_spacing * 12.0f, 1e-4f);
    // ...but with a floor on C0's COVERAGE for small scenes. The cascades are
    // camera-centred, so a framing viewpoint a scene-radius outside pushes the
    // far side of the scene past C0's 64-cell reach: on Cornell the back wall
    // fell 0.5 units outside and the screen gather returned pure sky for it, a
    // black rectangle right where the colour bleeding should be. Spanning twice
    // the diameter leaves room for that standoff. The cap keeps the rule from
    // firing on large scenes, where covering 2x the diameter at C0 would mean
    // metre-wide cells and no near field at all.
    {
        const float want = 2.0f * (2.0f * scene_bounds.radius()) / float(Config::kClipRes);
        cfg.cascade0_cell = std::max(cfg.cascade0_cell,
                                     std::min(want, cfg.surfel_spacing * 32.0f));
    }

    // Only build the cascades the scene needs. Cornell and Sponza both fit
    // inside C0 once its cell follows the surfel spacing, so the outer two would
    // otherwise aggregate the entire scene into a handful of cells.
    {
        // The cascades are CAMERA-centred, so covering the scene diameter is not
        // enough: a camera outside the scene pushes the far side out of range and
        // those surfels stop receiving updates entirely. Requiring the coarsest
        // cascade to span twice the diameter leaves room for the camera to sit
        // roughly a scene-radius outside and still cover everything.
        // Two constraints, both load-bearing:
        //
        //  - COVERAGE: the coarsest cascade must span twice the scene diameter,
        //    because the cascades are camera-centred and a camera outside the
        //    scene would otherwise push the far side out of range entirely.
        //
        //  - CELL SIZE: the gather visits every non-empty cell of the coarsest
        //    cascade, so that cascade has to be genuinely coarse. With a single
        //    cascade Sponza's "coarsest" cells are 0.23 units and there are
        //    ~10000 of them, which turned the far-field walk into 10000 cluster
        //    splats per surfel and cost 125 ms. Requiring the coarsest cell to be
        //    at least a tenth of the scene keeps that list in the low hundreds.
        const float diameter = 2.0f * scene_bounds.radius();
        cfg.cascades = 2;
        while (cfg.cascades < Config::kCascades &&
               (cfg.cascade_coverage(cfg.cascades - 1) < 2.0f * diameter ||
                cfg.cascade_cell(cfg.cascades - 1) < diameter * 0.10f))
            ++cfg.cascades;
        gllib::logf(gllib::LogLevel::info,
                    "cascades: %d active (C0 cell %.4f cover %.2f; coarsest cell %.4f "
                    "cover %.2f; scene diameter %.2f)",
                    cfg.cascades, cfg.cascade0_cell, cfg.cascade_coverage(0),
                    cfg.cascade_cell(cfg.cascades - 1),
                    cfg.cascade_coverage(cfg.cascades - 1), diameter);
    }

    gllib::logf(gllib::LogLevel::info,
                "scene surface area %.2f, LOD0 spacing %.4f (budget %u surfels)",
                scene_area, bake.world_spacing, kTargetLod0Surfels);

    SurfelLibrary surfels;
    surfels.build(instances, models, model_paths, bake);

    // --- Camera --------------------------------------------------------------
    gfx::Camera cam;
    cam.perspective(50.0f, float(window.width()) / float(window.height()),
                    std::max(0.01f, scene_bounds.radius() * 0.002f),
                    scene_bounds.radius() * 20.0f);
    cam.look_at(scene_bounds.center() +
                    (env.have_eye ? env.eye : scene.eye_rel) * scene_bounds.radius(),
                scene_bounds.center() +
                    (env.have_target ? env.target : scene.target_rel) * scene_bounds.radius());

    // --- Passes --------------------------------------------------------------
    GBuffer gbuf;
    gbuf.create(window.framebuffer_width(), window.framebuffer_height());

    GeometryPass geometry;
    DisplayPass display;
    SurfelDebugPass surfel_debug;
    Clipmap clipmap;
    CacheGI cache;
    ScreenGI screen_gi;
    EmitterSet emitters;
    if (!geometry.init() || !display.init() || !surfel_debug.init() || !clipmap.init() ||
        !screen_gi.init() || !cache.init(surfels.total_surfels())) {
        gllib::log(gllib::LogLevel::error, "shader initialisation failed");
        return 1;
    }
    // The G-buffer already exists at this point, so the loop's resize branch
    // never fires on frame 0 — size the screen-GI targets here instead.
    screen_gi.resize(gbuf.width, gbuf.height);

    // M5: fit emitter proxies to the baked emissive surfels. Static fit, then
    // re-transformed per frame for the movers.
    emitters.build(surfels, instances);

    // Sun + a single shadow map. Deliberately minimal: the cache integrates over
    // a hemisphere, which low-passes the shadow, so penumbra accuracy in the
    // bounce is wasted work. The pixel path gets CSM + PCSS in M5.
    gfx::ShadowMap sun_shadow(2048);
    glTextureParameteri(sun_shadow.handle(), GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTextureParameteri(sun_shadow.handle(), GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glTextureParameteri(sun_shadow.handle(), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(sun_shadow.handle(), GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    SunLight sun;
    // Cornell is lit entirely by its emissive ceiling panel, which is the point
    // of the scene; Sponza has no emissive materials at all, so it needs the sun.
    sun.enabled = (scene_idx == 1);

    // Gate 3: feed the gather a constant hemisphere of radiance L. The SH round
    // trip must reconstruct exactly L; anything else means the projection, the
    // solid angle or the cosine convolution is wrong.
    if (env.sh_test > 0.0f) {
        cache.set_debug_constant(true, env.sh_test);
        sun.enabled = false;
    }

    PassTimer t_frame("Frame", false);
    PassTimer t_gbuf("S7 G-buffer");
    PassTimer t_assemble("S0 assemble");
    PassTimer t_clipmap("S1 clipmap", false);   // GPU time comes from its sub-passes
    PassTimer t_display("Display");
    PassTimer t_shadow("Sun shadow");
    PassTimer t_points("Surfel debug");
    PassTimer t_imgui("ImGui");
    PassTimer* const timers[] = {&t_frame, &t_gbuf, &t_shadow, &t_assemble, &t_clipmap,
                                 &t_display, &t_points, &t_imgui};

    // --- State ---------------------------------------------------------------
    int view_mode = env.view;
    float exposure = env.exposure;
    bool animate = true;
    bool show_surfels = env.points;
    bool freeze_lod = false;
    float bounce_gain = 1.0f;
    // Artistic multiplier only. The physical intensity now arrives correctly
    // through KHR_materials_emissive_strength (Cornell's panel is authored at
    // 17), so this no longer has to compensate for a loader gap.
    float emissive_boost = 1.0f;
    if (env.emissive >= 0.0f) emissive_boost = env.emissive;
    float irradiance_gain = env.irradiance_gain;
    // Empty micro-buffer texels sample the sky. Sponza has no emissive materials
    // at all, so with a black sky its only light is the sun, which the roof
    // blocks almost everywhere -- the atrium reads as unlit. Cornell is a closed
    // box, so its sky is genuinely never sampled.
    glm::vec3 sky_radiance = (scene_idx == 1) ? glm::vec3(0.35f, 0.45f, 0.65f)
                                              : glm::vec3(0.0f);
    if (env.sky >= 0.0f) sky_radiance = glm::vec3(env.sky);
    // One cell, not two. The near field expands to INDIVIDUAL surfels, and the
    // surfel count inside a radius grows with its square: at radius 2 cells a
    // Sponza gather splatted ~1800 surfels into a 256-texel buffer, seven per
    // texel, for no added detail. The spec's budget is 300-600 splats total.
    int shell_radius = env.shell;
    int max_individual = 64;
    // Cone-march step counts. The cache gets the cheap end of the ladder on
    // purpose (spec section 2): a hemisphere integral low-passes the shadow, so
    // penumbra accuracy inside the bounce is wasted work.
    int cache_emitter_steps = env.esteps >= 0 ? env.esteps : 6;
    int pixel_emitter_steps = env.esteps >= 0 ? env.esteps : 16;
    int screen_search_radius = 0;
    float screen_reach = env.reach;
    float indirect_gain = 1.0f;
    if (env.spatial >= 0.0f) cache.spatial_strength() = env.spatial;
    if (env.budget > 0) cache.budget() = env.budget;
    float temporal_min = 0.05f;
    // 0.15, not the SVGF-ish 0.5.
    //
    // The micro-buffer's tangent frame is rotated per surfel AND per frame, so
    // successive refreshes are independent samples of the same integral rather
    // than a repeat of one deterministic estimate. That makes the temporal blend
    // a Monte Carlo average, and the variance-adaptive alpha was working against
    // it: it read the sampling jitter as "this surfel is still moving", pinned
    // alpha near its ceiling, and averaged about two frames. The result was
    // low-frequency blotching -- gather variance low-passed by S6 and S7 into
    // patches rather than removed. A lower ceiling averages ~7 refreshes and the
    // patches go.
    //
    // Latency is not the cost it looks like: S2's direct term is exact and
    // recomputed every frame, and a hard change still forces alpha to 1 through
    // the history reset. Only the multi-bounce tail converges over ~7 refreshes.
    float temporal_max = env.tmax >= 0.0f ? env.tmax : 0.15f;
    int budget_index = 2;
    int surfel_lod = std::clamp(env.surfel_lod, -1, Config::kSurfelLods - 1);
    int surfel_color = env.surfel_color;
    float point_scale = env.point_scale;
    bool captured = false;
    float sim_time = 0.0f;
    double last_time = window.time();
    double window_accum = 0.0;
    glm::mat4 prev_view_proj = cam.view_projection();
    std::vector<double> bench_ms;
    int frame_index = 0;

    while (!window.should_close()) {
        const double now = window.time();
        const double frame_ms = (now - last_time) * 1000.0;
        const float dt = float(std::min(now - last_time, 0.1));
        last_time = now;
        // Wall clock, not a CPU scope: see PassTimer::submit_external.
        if (frame_index > 0) t_frame.submit_external(frame_ms);

        window.poll_events();

        // Resize: the G-buffer rebuilds its (immutable) textures.
        const int fw = window.framebuffer_width();
        const int fh = window.framebuffer_height();
        if (fw > 0 && fh > 0 && (fw != gbuf.width || fh != gbuf.height)) {
            gbuf.create(fw, fh);
            screen_gi.resize(fw, fh);
            cam.set_aspect(float(fw) / float(fh));
        }

        camera_control(window, cam, dt, !gui.wants_mouse(), captured);

        // Hot reload.
        geometry.poll();
        display.poll();
        surfel_debug.poll();
        clipmap.poll();
        screen_gi.poll();
        if (cache.poll()) cache.reset_history();

        // Animate the rigid movers. prev_xform is kept so the G-buffer can emit
        // motion vectors and so M3's temporal filter has a reprojection basis.
        if (animate) sim_time += dt;
        for (size_t i = static_instance_count; i < instances.size(); ++i) {
            Instance& inst = instances[i];
            inst.prev_xform = inst.xform;
            const float a = sim_time * inst.orbit_speed + inst.orbit_phase;
            const glm::vec3 p = inst.orbit_center +
                glm::vec3(std::cos(a), 0.25f * std::sin(a * 1.7f), std::sin(a)) * inst.orbit_radius;
            inst.xform = glm::translate(glm::mat4(1.0f), p) *
                         glm::rotate(glm::mat4(1.0f), sim_time * inst.spin_speed, glm::vec3(0, 1, 0)) *
                         mover_base[i - static_instance_count];
        }
        // Proxies are fitted in object space, so a mover only needs its
        // rectangles re-transformed -- no refit, and the cached irradiance on
        // its surfels travels with it either way.
        if (animate) emitters.refresh(instances);

        const glm::mat4 view_proj = cam.view_projection();
        const glm::mat4 inv_view_proj = glm::inverse(view_proj);

        t_gbuf.begin();
        geometry.render(gbuf, instances, models, view_proj, prev_view_proj);
        t_gbuf.end();

        // Sun shadow map, re-rendered only when something moved.
        glm::mat4 sun_vp(1.0f);
        float shadow_texel = 0.0f;
        if (sun.enabled) {
            t_shadow.begin();
            sun_vp = gfx::compute_light_vp(view_proj, -glm::normalize(sun.direction),
                                           scene_bounds.radius(), scene_bounds.radius() * 2.0f);
            shadow_texel = scene_bounds.radius() * 2.0f / float(sun_shadow.size());
            sun_shadow.begin();
            for (const Instance& inst : instances) {
                if (inst.model < 0 || size_t(inst.model) >= models.size()) continue;
                const gfx::Model& m = *models[size_t(inst.model)];
                if (size_t(inst.mesh) >= m.mesh_count()) continue;
                sun_shadow.render_mesh(m.mesh(size_t(inst.mesh)), sun_vp * inst.xform);
            }
            sun_shadow.end();
            t_shadow.end();
        } else {
            t_shadow.skip();
        }

        // S0: object-space sets -> flat world-space live buffer.
        t_assemble.begin();
        const uint32_t live = clipmap.assemble(surfels, instances, cam, cfg, freeze_lod);
        t_assemble.end();

        // S2 must precede the cluster aggregation: clusters carry OUTGOING
        // radiance, which is what S2 computes.
        cache.direct_lighting(clipmap, sun, sun_vp,
                              sun.enabled ? sun_shadow.handle() : 0,
                              shadow_texel, bounce_gain, emissive_boost,
                              emitters, cache_emitter_steps);

        // S1: rebuild the cluster and occupancy cascades around the camera.
        t_clipmap.begin();
        clipmap.build(cam, cfg);
        t_clipmap.end();

        // S3 scheduler + S5 micro-render gather + S6 projection/temporal blend.
        cache.update_cache(clipmap, gbuf, cam, cfg, sky_radiance, shell_radius,
                           max_individual, temporal_min, temporal_max,
                           uint32_t(frame_index));

        // S7: gather the cache into a screen-space indirect buffer.
        screen_gi.render(clipmap, gbuf, cam, cache.sh_cache(), sky_radiance,
                         screen_search_radius, screen_reach);

        DisplayPass::Lighting lighting;
        lighting.indirect = &screen_gi.indirect();
        lighting.sun_shadow = sun.enabled ? sun_shadow.handle() : 0;
        lighting.sun_view_proj = sun_vp;
        lighting.sun_dir = sun.direction;
        lighting.sun_radiance = sun.color * sun.intensity;
        lighting.shadow_texel = shadow_texel;
        lighting.have_sun = sun.enabled;
        lighting.emissive_boost = emissive_boost;
        lighting.sky = sky_radiance;
        lighting.cam_pos = cam.position();
        lighting.emitters = &emitters;
        lighting.clip = &clipmap;
        lighting.emitter_steps = pixel_emitter_steps;
        lighting.indirect_gain = indirect_gain;

        t_display.begin();
        display.render(gbuf, view_mode, exposure, inv_view_proj,
                       scene_bounds.mn, scene_bounds.extent(), lighting);
        t_display.end();

        if (show_surfels) {
            t_points.begin();
            surfel_debug.render(surfels, instances, view_proj, gbuf,
                                surfel_lod, surfel_color, point_scale,
                                cam.near_clip(), cam.far_clip(),
                                &cache.sh_cache(), irradiance_gain,
                                &clipmap.instance_lod());
            t_points.end();
        } else {
            t_points.skip();
        }

        // --- UI ---------------------------------------------------------------
        t_imgui.begin();
        gui.begin_frame();
        if (env.no_gui) goto gui_done;
        ImGui::SetNextWindowSize(ImVec2(420, 0), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("MOSAIC")) {
            ImGui::Text("%s  |  %zu instances", scene.name, instances.size());
            {
                const double wall = t_frame.disp_cpu();
                const double gpu = t_gbuf.disp_gpu() + t_shadow.disp_gpu() +
                                   t_assemble.disp_gpu() + clipmap.gpu_ms() +
                                   cache.gpu_ms() + screen_gi.gpu_ms() +
                                   t_display.disp_gpu() +
                                   t_points.disp_gpu() + t_imgui.disp_gpu();
                ImGui::Text("frame %.2f ms (%.0f fps)   GPU %.2f ms",
                            wall, wall > 0.0 ? 1000.0 / wall : 0.0, gpu);
                if (gpu > wall * 1.25 && wall > 0.0)
                    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                                       "GPU exceeds wall clock: frames are queued ahead");
            }
            ImGui::Separator();

            int mode_count = 0;
            const char* const* mode_names = DisplayPass::view_mode_names(mode_count);
            ImGui::Combo("View", &view_mode, mode_names, mode_count);
            ImGui::SliderFloat("Exposure", &exposure, 0.05f, 8.0f, "%.2f",
                               ImGuiSliderFlags_Logarithmic);
            ImGui::Checkbox("Animate movers", &animate);

            ImGui::Separator();
            ImGui::Checkbox("Show surfels", &show_surfels);
            if (show_surfels) {
                int cc = 0;
                const char* const* cnames = SurfelDebugPass::color_mode_names(cc);
                ImGui::Combo("Surfel color", &surfel_color, cnames, cc);
                ImGui::SliderInt("Surfel LOD (-1 = live)", &surfel_lod, -1,
                                 Config::kSurfelLods - 1);
                ImGui::SliderFloat("Point scale", &point_scale, 0.2f, 6.0f, "%.2f");
            }

            if (ImGui::CollapsingHeader("Timing", ImGuiTreeNodeFlags_DefaultOpen)) {
                static const ImU32 cols[] = {
                    IM_COL32(220, 120,  60, 255),   // G-buffer
                    IM_COL32(120, 100,  80, 255),   // shadow
                    IM_COL32(230, 200,  70, 255),   // S0
                    IM_COL32(240, 140, 140, 255),   // S2
                    IM_COL32(190,  90, 200, 255),   // S1
                    IM_COL32(100, 200, 200, 255),   // S3
                    IM_COL32(250,  90,  90, 255),   // S5
                    IM_COL32(200, 140, 250, 255),   // S6
                    IM_COL32( 70, 220, 180, 255),   // S7 gather
                    IM_COL32(150, 230, 210, 255),   // S7 upsample
                    IM_COL32( 90, 170, 230, 255),   // display
                    IM_COL32(120, 210, 130, 255),   // points
                    IM_COL32(160, 160, 160, 255),   // imgui
                };
                const char* names[] = {t_gbuf.name(), t_shadow.name(), t_assemble.name(),
                                       cache.subpass(CacheGI::kDirect).name(),
                                       t_clipmap.name(),
                                       cache.subpass(CacheGI::kSchedule).name(),
                                       cache.subpass(CacheGI::kGather).name(),
                                       cache.subpass(CacheGI::kSpatial).name(),
                                       screen_gi.subpass(ScreenGI::kGather).name(),
                                       screen_gi.subpass(ScreenGI::kUpsample).name(),
                                       t_display.name(), t_points.name(), t_imgui.name()};
                const float vals[] = {float(t_gbuf.disp_gpu()),
                                      float(t_shadow.disp_gpu()),
                                      float(t_assemble.disp_gpu()),
                                      float(cache.subpass(CacheGI::kDirect).disp_gpu()),
                                      float(clipmap.gpu_ms()),
                                      float(cache.subpass(CacheGI::kSchedule).disp_gpu()),
                                      float(cache.subpass(CacheGI::kGather).disp_gpu()),
                                      float(cache.subpass(CacheGI::kSpatial).disp_gpu()),
                                      float(screen_gi.subpass(ScreenGI::kGather).disp_gpu()),
                                      float(screen_gi.subpass(ScreenGI::kUpsample).disp_gpu()),
                                      float(t_display.disp_gpu()),
                                      float(t_points.disp_gpu()),
                                      float(t_imgui.disp_gpu())};
                constexpr int kPassCount = 13;
                float total = 0.0f;
                for (float v : vals) total += v;
                const ImVec2 p = ImGui::GetCursorScreenPos();
                imgui_stacked_bar(p, ImVec2(ImGui::GetContentRegionAvail().x, 14),
                                  vals, cols, kPassCount);
                ImGui::Dummy(ImVec2(0, 18));
                imgui_stacked_legend("passes", names, vals, cols, kPassCount, total);
                ImGui::Text("GPU total: %.3f ms", total);
            }

            if (ImGui::CollapsingHeader("Surfels", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("%u surfels in %zu sets (%u cached)",
                            surfels.total_surfels(), surfels.entries().size(),
                            surfels.cached_sets());
                ImGui::Text("%u emissive, bake %.2f s",
                            surfels.total_emissive(), surfels.bake_seconds());
                ImGui::Text("%.1f MB packed", double(surfels.total_surfels()) * 16.0 / 1048576.0);
                // Reported in WORLD units. The per-set values are object-space,
                // and Sponza's fit scale is ~0.0015, so raw set numbers are not
                // comparable between meshes, let alone between scenes.
                if (ImGui::BeginTable("lods", 4)) {
                    ImGui::TableSetupColumn("LOD");
                    ImGui::TableSetupColumn("Count");
                    ImGui::TableSetupColumn("Spacing (world)");
                    ImGui::TableSetupColumn("Mean r (world)");
                    ImGui::TableHeadersRow();
                    for (int l = 0; l < Config::kSurfelLods; ++l) {
                        uint32_t count = 0;
                        double r_sum = 0.0;
                        for (const auto& e : surfels.entries()) {
                            count += e.set.lods[l].count;
                            r_sum += double(e.set.lods[l].mean_radius) *
                                     double(e.object_scale) * double(e.set.lods[l].count);
                        }
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("%d", l);
                        ImGui::TableNextColumn(); ImGui::Text("%u", count);
                        ImGui::TableNextColumn();
                        ImGui::Text("%.4f", bake.world_spacing * Config::kLodSpacing[l]);
                        ImGui::TableNextColumn();
                        ImGui::Text("%.4f", count ? r_sum / double(count) : 0.0);
                    }
                    ImGui::EndTable();
                }
            }

            if (ImGui::CollapsingHeader("Clipmap", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("%u live surfels (cap %u)", live, Config::kMaxLiveSurfels);
                ImGui::Checkbox("Freeze LOD at 0", &freeze_lod);
                if (ImGui::BeginTable("cascades", 4)) {
                    ImGui::TableSetupColumn("Cascade");
                    ImGui::TableSetupColumn("Cell");
                    ImGui::TableSetupColumn("Coverage");
                    ImGui::TableSetupColumn("Non-empty cells");
                    ImGui::TableHeadersRow();
                    for (int l = 0; l < cfg.cascades; ++l) {
                        const Cascade& c = clipmap.cascade(l);
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("C%d", l);
                        ImGui::TableNextColumn(); ImGui::Text("%.4f", c.cell);
                        ImGui::TableNextColumn();
                        ImGui::Text("%.2f", c.cell * float(Config::kClipRes));
                        ImGui::TableNextColumn(); ImGui::Text("%u", clipmap.occupied_cells(l));
                    }
                    ImGui::EndTable();
                }
                if (ImGui::BeginTable("s1sub", 2)) {
                    for (int i = 0; i < Clipmap::kSubPassCount; ++i) {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("S1 %s", Clipmap::subpass_name(i));
                        ImGui::TableNextColumn();
                        ImGui::Text("%.3f ms", clipmap.subpass(i).disp_gpu());
                    }
                    ImGui::EndTable();
                }
                int lod_hist[Config::kSurfelLods] = {};
                int culled = 0;
                for (int l : clipmap.instance_lod()) {
                    if (l < 0) ++culled; else ++lod_hist[l];
                }
                ImGui::Text("instance LOD: %d / %d / %d / %d  (%d culled)",
                            lod_hist[0], lod_hist[1], lod_hist[2], lod_hist[3], culled);
                if (clipmap.budget_dropped() > 0)
                    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                                       "%u instance(s) dropped: live budget exhausted",
                                       clipmap.budget_dropped());
            }

            if (ImGui::CollapsingHeader("Cache (S2-S6)", ImGuiTreeNodeFlags_DefaultOpen)) {
                static const uint32_t kBudgets[] = {2048, 4096, 8192, 16384, 32768};
                static const char* kBudgetNames[] = {"2k", "4k", "8k", "16k", "32k"};
                if (ImGui::Combo("Update budget", &budget_index, kBudgetNames, 5))
                    cache.budget() = kBudgets[budget_index];
                ImGui::Text("%u surfels updated this frame", cache.selected_count());
                ImGui::Text("refresh: every ~%.1f frames",
                            cache.budget() > 0 ? double(live) / double(cache.budget()) : 0.0);

                ImGui::SliderInt("Shell radius (cells)", &shell_radius, 1, 4);
                ImGui::SliderInt("Max individual per cell", &max_individual, 4, 512);
                ImGui::SliderInt("Screen gather radius", &screen_search_radius, 0, 2);
                ImGui::SliderFloat("Screen gather reach", &screen_reach, 1.0f, 6.0f, "%.2f r");
                ImGui::SliderFloat("Indirect gain", &indirect_gain, 0.0f, 8.0f, "%.2f");
                ImGui::SliderFloat("Bounce gain", &bounce_gain, 0.0f, 3.0f, "%.2f");
                ImGui::SliderFloat("Emissive boost", &emissive_boost, 0.0f, 200.0f, "%.1f");
                ImGui::SliderFloat("Irradiance view gain", &irradiance_gain,
                                   0.05f, 20.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("Spatial filter", &cache.spatial_strength(), 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Temporal alpha min", &temporal_min, 0.01f, 1.0f, "%.3f");
                ImGui::SliderFloat("Temporal alpha max", &temporal_max, 0.01f, 1.0f, "%.3f");
                ImGui::ColorEdit3("Sky radiance", &sky_radiance.x,
                                  ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
                if (ImGui::Button("Reset cache history")) cache.reset_history();

                ImGui::Separator();
                ImGui::Checkbox("Sun", &sun.enabled);
                if (sun.enabled) {
                    ImGui::SliderFloat3("Sun dir", &sun.direction.x, -1.0f, 1.0f);
                    ImGui::ColorEdit3("Sun color", &sun.color.x);
                    ImGui::SliderFloat("Sun intensity", &sun.intensity, 0.0f, 20.0f);
                }
            }

            if (ImGui::CollapsingHeader("Config")) {
                ImGui::Text("Cascade 0 cell:  %.4f", cfg.cascade0_cell);
                ImGui::Text("Surfel spacing:  %.4f", cfg.surfel_spacing);
                ImGui::Text("Clipmap:         %d^3 x %d cascades",
                            Config::kClipRes, Config::kCascades);
                ImGui::Text("Scan limit:      %u elements", PrefixSum::kMaxElements);
            }
        }
        ImGui::End();
    gui_done:
        gui.render();
        t_imgui.end();

        window.swap_buffers();

        // Read the previous frame's queries; never blocks on a live query.
        for (PassTimer* t : timers) t->readback();
        for (int i = 0; i < Clipmap::kSubPassCount; ++i) clipmap.subpass(i).readback();
        for (int i = 0; i < CacheGI::kSubPassCount; ++i) cache.subpass(i).readback();
        for (int i = 0; i < ScreenGI::kSubPassCount; ++i) screen_gi.subpass(i).readback();
        // Startup frames (shader compilation, surfel bake, driver clock ramp) are
        // wildly slower than steady state and would otherwise dominate the first
        // displayed average, which is the one a short run ever shows.
        if (frame_index < 30) {
            for (PassTimer* t : timers) t->flush_window();
            for (int i = 0; i < Clipmap::kSubPassCount; ++i) clipmap.subpass(i).flush_window();
            for (int i = 0; i < CacheGI::kSubPassCount; ++i) cache.subpass(i).flush_window();
            for (int i = 0; i < ScreenGI::kSubPassCount; ++i) screen_gi.subpass(i).flush_window();
            window_accum = 0.0;
        }

        window_accum += frame_ms;
        if (window_accum >= 500.0) {
            for (PassTimer* t : timers) t->flush_window();
            for (int i = 0; i < Clipmap::kSubPassCount; ++i) clipmap.subpass(i).flush_window();
            for (int i = 0; i < CacheGI::kSubPassCount; ++i) cache.subpass(i).flush_window();
            window_accum = 0.0;
        }

        // Validation runs once, a few frames in, so the pipeline is in its
        // steady state. It stalls hard (six full buffer read-backs), so it is
        // never on the normal path.
        if (env.validate && frame_index == 3) clipmap.log_instances_once();
        if (env.validate && frame_index == 4) clipmap.validate(cfg);
        if (env.validate && (frame_index == 60 || frame_index == 240)) cache.log_stats(&surfels);

        prev_view_proj = view_proj;
        ++frame_index;

        if (env.bench > 0) {
            if (frame_index > 0) bench_ms.push_back(frame_ms);
            if (frame_index >= env.bench) break;
        }
    }

    if (env.shot) gfx::screenshot(env.shot_path.c_str());
    if (env.bench > 0) {
        report_bench(std::move(bench_ms));
        // Per-pass GPU averages over the whole run, so a bench is enough to
        // locate a regression without opening the UI.
        printf("[bench] GPU per pass (avg ms):\n");
        double total = 0.0;
        for (PassTimer* t : timers) {
            if (!t->gpu()) continue;
            printf("    %-16s %7.3f\n", t->name(), t->avg_gpu());
            total += t->avg_gpu();
        }
        for (int i = 0; i < Clipmap::kSubPassCount; ++i) {
            printf("    S1 %-13s %7.3f\n", Clipmap::subpass_name(i),
                   clipmap.subpass(i).avg_gpu());
            total += clipmap.subpass(i).avg_gpu();
        }
        for (int i = 0; i < CacheGI::kSubPassCount; ++i) {
            printf("    %-16s %7.3f\n", CacheGI::subpass_name(i),
                   cache.subpass(i).avg_gpu());
            total += cache.subpass(i).avg_gpu();
        }
        for (int i = 0; i < ScreenGI::kSubPassCount; ++i) {
            printf("    %-16s %7.3f\n", ScreenGI::subpass_name(i),
                   screen_gi.subpass(i).avg_gpu());
            total += screen_gi.subpass(i).avg_gpu();
        }
        printf("    %-16s %7.3f\n", "GPU TOTAL", total);
    }

    gui.shutdown();
    return 0;
}

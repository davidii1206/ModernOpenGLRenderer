#pragma once

// ---------------------------------------------------------------------------
// Variant A, brute force: the recursive camera schedule.
//
// Design doc section 6.1's structure, with the camera counts made explicit:
//
//   Pass 0  primary G-buffer                    (hardware raster, screen.cpp)
//   Pass 1  a camera at every GI-grid pixel     N        cameras
//   Pass 2  a camera per tile of every pass-1 target   N * K1
//   Pass 3  a camera per tile of every pass-2 target   N * K1 * K2
//   Resolve deepest level -> ... -> level 1 -> the GI image
//
// WHY THE SCHEDULE IS CHUNKED. The N^3 in the doc's title for this variant is
// not rhetorical. At 1600x900 with MBG_SCALE=4 there are 90k level-1 cameras; at
// the default 16 tiles per camera that is 1.4M level-2 cameras and 5.8M at
// level 3. Their buffers alone would be ~300 MB and the frame would take
// seconds. So the sweep is split: MBG_BUDGET level-1 cameras per frame, each one
// carrying its whole subtree, with the GI image persisting between frames. The
// image is complete after ceil(pixels / budget) frames and the cursor then
// wraps. This is progressive rendering, NOT temporal feedback -- nothing reads
// last frame's radiance, which is exactly what separates this variant from
// variant B and is why it is correct on its first complete sweep.
//
// WHAT IS DELIBERATELY MISSING. No camera clustering (section 4.1), no cluster
// DAG LOD (section 5.2), no work list or vertex amortization (section 5.4), no
// Russian roulette or importance-based spawning (section 6.2 beyond resolution
// falloff and tile clustering), no radiance cache (section 7). Section 8.2
// orders those after this, and every one of them is an approximation that this
// example is supposed to be the yardstick for.
// ---------------------------------------------------------------------------

#include "gpu_util.hpp"
#include "scene.hpp"
#include "screen.hpp"
#include "sky.hpp"

#include <array>

namespace mbg {

// 3: the doc's variant A terminates at bounce 3 and that is where this renderer
// is being run, so it is where the schedule stops.
//
// THIS CONSTANT IS THE ONLY THING HOLDING IT THERE. The path estimator's cost is
// paths x depth (finding 20), so raising this is affordable in a way it never
// was under the branching recursion -- and finding 20's own tables, which go to
// eight bounces, were measured with it at 8. Raise it and the arrays below
// extend with 8x8 targets; nothing else needs to change. What does NOT survive
// the raise is the `paths` gate's level counts and the ImGui slider's range,
// both of which are written against this number rather than against a literal.
constexpr uint32_t kMaxLevels = 3;

struct SolveConfig {
    uint32_t bounces = 3;          // camera levels; 1 == direct only
    // GI grid = framebuffer / scale.
    //
    // 4, and finding 21 is the record of trying to make it finer and putting it
    // back. A finer grid does resolve the crease it was supposed to -- and it
    // makes SILHOUETTES worse, because scale 4's softness was the only thing
    // hiding the upsample's per-block structure along a diagonal edge. Trading a
    // soft crease for a stepped silhouette is a bad trade, and it is one that no
    // aggregate metric in this file reports.
    uint32_t scale   = 4;
    uint32_t budget  = 4096;       // level-1 cameras per frame

    // The resolution ladder of section 6.2 / 4.1: near cameras get the
    // resolution, deep ones do not, because deep bounces carry little energy and
    // are extremely low frequency.
    //
    // 16 everywhere, which is HALF the doc's near tier and a sixteenth of the
    // directions this example briefly thought it needed. Throwing resolution at
    // the indirect term was treating a correlation problem as a sampling-rate
    // problem: once each receiver's frame is rotated and the grid is denoised,
    // 16x16, 32x32 and 64x64 produce the same image to within 0.0001 RMSE.
    // See implementation.md, finding 14.
    std::array<uint32_t, kMaxLevels> res{{16, 8, 8}};
    // Tile edge for spawning the next level, in texels of THIS level's target.
    // 1 spawns per texel (the unabridged recursion); the terminal level ignores
    // it. res/block must be an integer, and configure() snaps it down until it
    // is -- a tile edge that does not divide the target would drop the remainder.
    //
    // 8 at level 1, on measurement. Before the direct term was split out of the
    // clustered mass (implementation.md, finding 4) a tile of 8 was visibly the
    // worst setting here -- hard-edged panel-shaped patches that RMSE barely
    // registered. With the split it is worth 0.0533 against tile 4's 0.0531 and
    // tile 2's 0.0529, for a sixth and a sixtieth of the time respectively.
    //
    // The deeper levels keep 4 because their targets are 8x8: a tile of 8 there
    // spawns a single child and the level below it stops being a gather at all,
    // and the extra children are cheap next to level 1's, which multiplies
    // everything under it.
    // 4 at a 16-wide target is 16 children, which is where finding 4's
    // measurement put it -- that finding is about how many children there are,
    // not how many texels each covers.
    std::array<uint32_t, kMaxLevels> block{{4, 4, 4}};

    // --- The estimator -------------------------------------------------------
    //
    // SINGLE-SAMPLE CONTINUATION. Split once at the primary hit into `paths`
    // directions, then let each path pick exactly ONE continuation per bounce.
    // Branch factor 1, so a depth-D solve costs paths x D cameras instead of
    // the branching recursion's K^D -- and every knob below that shrinks K only
    // shrinks the base of that exponent, which is why this is the setting that
    // actually makes depth affordable.
    //
    // 0 selects the branching tile estimator instead: K = (res/block)^2 children
    // per camera, each carrying a whole tile of directions scaled by one
    // representative's visibility. That one is deterministic, which is why every
    // gate runs on it, and it is the doc's variant A as written.
    //
    // 20, BECAUSE THE ESTIMATOR SATURATES FAR BELOW THE EQUAL-COST POINT.
    //
    // The equal-cost point is 40: at the three-bounce cap the branching tree
    // spawns 16 then 4, so a primary hit carries 1 + 16 + 64 = 81 cameras, and a
    // path split of S carries 1 + 2S with 1 + 2*40 = 81. Same cameras, same
    // texels, same triangle-rasters, to the unit. That is the comparison finding
    // 20 is written against and MBG_PATHS=40 still reproduces it exactly.
    //
    // But the split saturates long before it: 20, 40, 64 and 256 all score the
    // same RMSE to three digits (0.0403 / 0.0402 / 0.0402 / 0.0402), the same
    // whole-image high-pass noise (3.87 against 3.88) and the same ceiling noise
    // (1.39 against 1.40). Between 20 and 40 the largest single-pixel difference
    // in the reference frame is 17/255 and only 99 pixels of 262144 differ by
    // more than 8 -- and, checked the way finding 24 says to check a default,
    // the ceiling/wall diagonal and the vertical corner are step-for-step
    // identical. Halving the split is 1.63x measured end to end (22990 ms to
    // 14094 ms, 1.33e6 cameras to 6.7e5) for an image nobody can pick out of a
    // line-up. Cost is 1 + paths*(bounces-1) cameras per primary hit, so this is
    // also what makes a fourth bounce cost what three used to.
    uint32_t  paths = 20;
    // Russian roulette threshold on the path throughput (the running product of
    // what each bounce reflects, near enough). Below it a path survives with
    // probability throughput/threshold and is divided by that probability, so
    // the estimate stays unbiased and the expected path length becomes finite
    // and scene-dependent rather than pinned to `bounces`. 0 disables it.
    //
    // 0.15 never fires within three bounces off Cornell's walls (0.73^3 = 0.39),
    // so at the current kMaxLevels it is inert by construction and the solve is
    // exactly the solve it would be without it. It is kept, and gated, because
    // it costs nothing when dormant and it is the thing that makes raising
    // kMaxLevels safe -- without it, depth is unbounded work for energy that is
    // already below the tone curve's resolution.
    float     rr = 0.15f;
    // Let the terminal level answer "is this direction blocked" instead of
    // "by what" -- exact there, and the distance bound then terminates the
    // search almost immediately. 0 ablates it. See raster.glsl.
    bool      anyhit = true;
    // Reject a cluster no live direction passes through, before fetching its 64
    // triangles. The distance bound asks whether a box is near enough; this asks
    // whether it is in the way, which on a scene with clusters is the question
    // that removes almost everything. 0 ablates it. See raster.glsl.
    bool      angular = true;
    // The SAME test in the light view's occlusion loop, which is a second
    // traversal that never got any of the hemisphere's work. Separate from
    // `angular` so the two can be attributed apart -- they have different
    // conservativeness, because lv_tri_hit carries finding 19's half-space rule
    // and that reports a hit for rays which miss the triangle geometrically.
    bool      lv_angular = true;
    // Draw the continuation direction from the micro-buffer's own radiance --
    // cos * dOmega * albedo * (unshadowed direct irradiance at the hit) --
    // rather than from cos * dOmega * albedo alone. The extra factor is the only
    // guess in it, and it costs nothing: the direct irradiance at every texel's
    // hit point is already evaluated for the mass. 0 ablates it.
    bool      importance = true;
    // Cull triangle clusters against each light view's frustum before walking
    // them. 0 walks the whole scene, which is the measurement this is compared
    // against. See scene.hpp's GpuCluster and lightview.glsl's lv_cull.
    bool      cull = true;
    // One traversal per workgroup with several texels per thread, instead of
    // one traversal per texel. See mbg_resolve_vis_coop.
    bool      coop = true;
    // Skip the per-pixel light view where the GI grid's own visible fractions
    // already agree across the block, and take the fraction from them instead.
    // See direct_pixel.comp. The thresholds are per light character, and the
    // sun's is tighter because its penumbra is narrow.
    bool      direct_mask = true;
    // In TEXELS of the grid's own light view, not in absolute fraction: that
    // view is cam_lv_res^2 texels, so its answer is quantised to 1/that, and a
    // threshold below the quantum can never be met. See direct_pixel.comp.
    float     mask_texels_e = 2.0f;
    float     mask_texels_s = 1.0f;
    // Visit the coarse groups nearest-first, so the per-texel distance bound
    // tightens early instead of at whatever point the Morton order happens to
    // cross the receiver's end of the scene. See mbg_order_groups.
    bool      order = true;

    // The direct term. `nee` routes every emissive triangle through the analytic
    // estimator in raster.comp instead of letting the quadrature find it, which
    // is doc section 3's "direct lighting stays a separate conventional pass"
    // and is worth more to the image than any other single setting here.
    bool      nee = true;
    // Rasterize a hemisphere PER PIXEL for the image's direct term instead of
    // taking it from the GI grid. The grid answers for a scale x scale block, so
    // the direct term -- the sharpest thing in the image -- is otherwise limited
    // by the grid's Nyquist and comes out soft however good the upsample is.
    // Same rasterizer and same depth sort as everywhere else; only the receiver
    // changes. The solve is unchanged; only what the image reads changes.
    bool      direct_pixel = true;
    // Edge of the per-pixel pass's light view. 8 puts 64 texels on the emitter
    // -- against the 4-29 a 32x32 hemisphere manages, which is the whole point
    // of fitting the frustum (common/lightview.glsl). Measured indistinguishable
    // from 16 and 32 in both RMSE and roughness, at a quarter of the cost, and
    // this pass is the most expensive thing in a frame.
    uint32_t  direct_res = 8;
    // Light-view edge for the SECONDARY CAMERAS' direct term. Smaller than the
    // per-pixel pass's because it feeds bounce transport rather than the image,
    // but it cannot be the hemisphere: at an 8x8 target a camera's hemisphere
    // puts half a texel on Cornell's panel, and a yes/no answer there is the
    // banding the indirect term used to show.
    uint32_t  cam_lv_res = 8;
    // Composite nothing but the indirect residual: the solve runs exactly as it
    // does normally and the per-pixel direct pass is skipped, so the image is
    // the bounce term alone. A diagnostic, not a rendering mode -- the indirect
    // is usually 10x dimmer than the direct and invisible underneath it.
    bool      indirect_only = false;
    // Rotate every receiver's tangent frame by a hash of its position, so the
    // quadrature error decorrelates between neighbours instead of banding.
    bool      jitter = true;
    // A-trous iterations over the GI grid, and taps per side. Removes what the
    // jitter turned into noise; safe only because the grid carries the indirect
    // residual alone. 0 disables.
    //
    // 2, not 3, on measurement. Iteration 3 has a tap spacing of 4 cells and so
    // reaches 8 cells out -- a third of the way across the ceiling, which is only
    // about 23 grid rows tall at this scale because it is seen nearly edge on.
    // At that reach the filter is no longer averaging neighbours that share a
    // neighbourhood, and what it produces is row structure: with the sun on, the
    // ceiling's high-pass noise goes from isotropic at 2 iterations (row/col
    // 1.11) to plainly horizontal at 3 (1.62), for 0.0005 of RMSE.
    uint32_t  filter_iters = 2;
    int32_t   filter_radius = 2;
    // Spread each texel's albedo mass across the four nearest spawn tiles
    // instead of assigning it to one. Removes the tile discontinuity; costs a
    // wider scan of shared memory and nothing else.
    bool      tent = true;

    float     bias = 1e-3f;        // camera offset along its own normal, world units
    // The environment: what an uncovered texel sees, and the one light in the
    // scene that has no geometry. All zero by default, so every measurement in
    // implementation.md and every gate reads exactly what it did before this
    // existed. See sky.hpp and shaders/common/sky.glsl.
    SkyLight  sky{};
    float     emissive = 1.0f;
    bool      two_sided = false;
    float     plane_tol = 0.05f;   // upsample plane cutoff, world units
    bool      running = true;

    // TALLY WHAT THE TRAVERSAL ACTUALLY TOUCHED. Off by default: it costs a
    // workgroup-uniform branch per loop iteration and one atomic per counter per
    // camera, which is nothing, but "nothing" is a claim this example is not in
    // a position to verify on llvmpipe (finding 22), so the shipped path does
    // not carry it. See shaders/common/counters.glsl and MBG_COUNT.
    bool      count = false;
    // Bracket the kernel's five phases with GL_ARB_shader_clock, so a dispatch
    // that reads 3 s can say WHICH of the five it spent it in. Off by default
    // for the same reason as `count`. See shaders/common/perf.glsl.
    bool      perf = false;

    // Only the fields that change buffer sizes.
    bool layout_equals(const SolveConfig& o) const {
        if (bounces != o.bounces || scale != o.scale || budget != o.budget ||
            paths != o.paths) return false;
        for (uint32_t i = 0; i < bounces; ++i)
            if (res[i] != o.res[i] || block[i] != o.block[i]) return false;
        return true;
    }
};

struct LevelInfo {
    uint32_t res = 0;
    uint32_t block = 0;        // 0 on the terminal level
    uint32_t children = 0;     // tiles per camera = (res/block)^2
    uint32_t cameras = 0;      // per chunk
    std::size_t bytes = 0;
};

class Solver {
public:
    bool init();
    bool poll();

    // (Re)allocates when the layout or the GI grid changed; otherwise a no-op.
    void configure(const SolveConfig& cfg, int gi_w, int gi_h);
    // Restarts the sweep. The GI image keeps its contents so a restart shows
    // stale-but-plausible lighting rather than black.
    void restart() { cursor_ = 0; }
    void clear_image();

    // One chunk: place, rasterize every level, resolve back up, scatter.
    void step(const GBuffer& gb, const gfx::Camera& cam, const Scene& scene,
              const SolveConfig& cfg);
    // GI grid -> full resolution. Runs every frame, independent of the chunk.
    void upsample(const GBuffer& gb, const gfx::Camera& cam, const SolveConfig& cfg);
    // Adds the per-pixel direct term into the upsampled target. Runs after
    // upsample(), every frame, and is independent of the chunk schedule.
    void direct_pixel(const GBuffer& gb, const gfx::Camera& cam, const Scene& scene,
                      const SolveConfig& cfg);
    // Denoise the GI grid in place, before upsampling. Display only.
    void filter(const GBuffer& gb, const gfx::Camera& cam, const SolveConfig& cfg);

    // --- Gate entry points ---------------------------------------------------
    //
    // The same kernels driven from an explicit camera list instead of a
    // G-buffer, so the analytic gates in validate.cpp measure the shipping code
    // path rather than a reimplementation of it. Both reallocate the level
    // buffers for `cams.size()` cameras and leave the solver needing a
    // configure() before the next frame.

    // Irradiance per camera, the whole recursion.
    std::vector<glm::vec4> solve_points(const Scene& scene, const SolveConfig& cfg,
                                        const std::vector<glm::vec4>& pos,
                                        const std::vector<glm::vec4>& nrm);
    // Level-1 visibility keys, cams.size() * res*res of them, for comparing the
    // compute rasterizer against a CPU ray-cast oracle (doc section 8.2
    // milestone 3).
    std::vector<uint32_t> raster_visibility(const Scene& scene, const SolveConfig& cfg,
                                            const std::vector<glm::vec4>& pos,
                                            const std::vector<glm::vec4>& nrm,
                                            uint32_t mode = 1);

    const gl::Texture& target() const { return full_; }

    // --- Traversal work counters (SolveConfig::count) -----------------------
    //
    // Zero them, run whatever is being measured, then read them back. The
    // readback maps the buffer and so synchronizes; it is a measurement tool
    // and belongs nowhere near a frame.
    void count_reset();
    struct Counts {
        double cameras = 0, grp_test = 0, grp_enter = 0, clu_test = 0,
               clu_enter = 0, tri_setup = 0, tex_test = 0, texels = 0,
               clu_ang = 0, tex_ang = 0, clu_pair = 0, ang_viol = 0,
               lv_pair = 0, lv_ang = 0;
    };
    Counts count_read() const;

    // --- In-shader phase cycles (SolveConfig::perf) --------------------------
    //
    // Cycles, not milliseconds, and only comparable WITHIN one dispatch: the
    // clock counts issue cycles on whatever unit ran the invocation. Read every
    // half second rather than every frame -- the readback synchronizes, and two
    // stalls a second on a diagnostic that is off by default is a fair trade for
    // not double-buffering a 28-byte buffer.
    void perf_reset();
    struct Phases {
        double cameras = 0;
        double cyc[7] = {};      // setup traverse quad emitter sun reduce spawn
        double pixels = 0;       // direct_pixel.comp's own normalizer
        double dp[4] = {};       // dp: setup, mask, emitter LV, sun LV
        static const char* name(int i);
    };
    Phases perf_read() const;

    uint32_t cursor() const { return cursor_; }
    uint32_t sweeps() const { return sweeps_; }
    uint32_t gi_pixels() const { return uint32_t(gi_w_ * gi_h_); }
    glm::ivec2 gi_size() const { return glm::ivec2(gi_w_, gi_h_); }
    uint32_t levels() const { return levels_; }
    const LevelInfo& level(uint32_t i) const { return info_[i]; }
    std::size_t bytes() const;
    // Cameras and rasterized texels in the chunk just submitted, for the cost
    // model in the ImGui panel and in --bench.
    double chunk_cameras() const;
    double chunk_texels() const;
    // The same totals for a COMPLETE sweep of the GI grid, which is the unit a
    // frame time should be compared against: a chunk is an arbitrary slice, a
    // sweep is one finished image.
    double sweep_cameras() const;
    double sweep_texels() const;

    PassTimer& t_place() { return t_place_; }
    PassTimer& t_raster(uint32_t l) { return t_raster_[l]; }
    PassTimer& t_gather() { return t_gather_; }
    PassTimer& t_upsample() { return t_upsample_; }
    PassTimer& t_direct() { return t_direct_; }
    PassTimer& t_filter() { return t_filter_; }

private:
    void dispatch_1d(uint32_t count);          // 64-wide, split across x/y
    void raster_level(uint32_t l, uint32_t count, const Scene& scene,
                      const SolveConfig& cfg, gl::Buffer* dump);
    // Rasterize every level for `chunk` level-1 cameras, then resolve back up.
    // Shared by step() and the gate entry points.
    void run_levels(uint32_t chunk, const Scene& scene, const SolveConfig& cfg,
                    bool write_image);
    void upload_points(const std::vector<glm::vec4>& pos, const std::vector<glm::vec4>& nrm);

    Pipeline place_, gather_, upsample_, direct_px_, filter_;
    // ONE KERNEL PER TARGET SIZE. s_vis and s_cdf are sized by MBG_MAX_TEXELS,
    // and finding 38 measured 4 KB of shared memory costing 1.9x on this kernel,
    // so a single variant sized for the largest target makes every level pay for
    // the largest. The default schedule's levels 2 and 3 use an 8x8 target and
    // carry 40 of its 41 cameras; they need 512 bytes and were reserving 8 KB.
    // Ascending capacity; raster_for() picks the smallest that fits.
    static constexpr uint32_t kRasterTiers = 3;
    static constexpr uint32_t kTierTexels[kRasterTiers] = {64, 256, 1024};
    std::array<Pipeline, kRasterTiers> raster_;
    const Pipeline& raster_for(uint32_t res) const;
    Pipeline& raster_for(uint32_t res);

    // Per level: cameras, irradiance, the direct term with its visible fraction,
    // and the per-tile masses that spawned them (two vec4s per child: the
    // indirect mass and the direct mass -- see raster.comp).
    std::array<gl::Buffer, kMaxLevels> cams_, irrad_, direct_, weights_;
    std::array<Quadrature, kMaxLevels> quad_;
    Quadrature direct_quad_;       // for the per-pixel direct pass
    std::array<LevelInfo, kMaxLevels> info_{};
    uint32_t levels_ = 0;

    // Seven 64-bit counters as pairs of 32-bit words; see counters.glsl.
    mutable gl::Buffer counters_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    mutable gl::Buffer perf_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};

    gl::Texture gi_{gl::TextureType::tex_2d};      // GI grid, RGBA32F, (E, 1)
    gl::Texture gi_tmp_{gl::TextureType::tex_2d};  // ping-pong for the denoise
    // Per GI cell: (emitter visible fraction, sun visible fraction, 0, written).
    // The grid already MEASURES both -- finding 18 keeps them apart because an
    // emitter's penumbra is soft and a sun's is not -- so this texture only
    // carries numbers that already existed. See direct_pixel.comp's mask.
    gl::Texture vis_{gl::TextureType::tex_2d};
    // The denoised copy. SEPARATE FROM gi_ ON PURPOSE: the grid accumulates
    // across frames (a sweep is chunked), so a filter that wrote back into it
    // would re-filter every cell the chunk cursor is not currently rewriting,
    // once per frame, forever. That is not "3 a-trous iterations", it is an
    // unbounded number of them -- and what survives unbounded a-trous is
    // whatever its dilated passes cannot attenuate, which is the grid's own
    // Nyquist. See implementation.md, finding 17.
    gl::Texture gi_disp_{gl::TextureType::tex_2d};
    gl::Texture full_{gl::TextureType::tex_2d};    // framebuffer resolution
    int gi_w_ = 0, gi_h_ = 0, full_w_ = 0, full_h_ = 0;

    SolveConfig layout_{};
    bool allocated_ = false;
    uint32_t cursor_ = 0, sweeps_ = 0, last_chunk_ = 0;
    uint32_t dump_mode_ = 1;

    PassTimer t_place_{"Place"};
    std::array<PassTimer, kMaxLevels> t_raster_{
        PassTimer{"Raster L1"}, PassTimer{"Raster L2"}, PassTimer{"Raster L3"}};
    PassTimer t_gather_{"Resolve"};
    PassTimer t_upsample_{"Upsample"};
    PassTimer t_direct_{"Direct/px"};
    PassTimer t_filter_{"GI filter"};
};

} // namespace mbg

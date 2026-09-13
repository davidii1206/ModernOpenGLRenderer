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

#include <array>

namespace mbg {

// 4 is arbitrary but not accidental: the doc's variant A terminates at bounce 3,
// and one more level exists so that "does the 4th bounce change anything" is a
// question this example can answer rather than assume.
constexpr uint32_t kMaxLevels = 4;

struct SolveConfig {
    uint32_t bounces = 3;          // camera levels; 1 == direct only
    uint32_t scale   = 4;          // GI grid = framebuffer / scale
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
    std::array<uint32_t, kMaxLevels> res{{16, 8, 8, 8}};
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
    std::array<uint32_t, kMaxLevels> block{{4, 4, 4, 4}};

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
    uint32_t  filter_iters = 3;
    int32_t   filter_radius = 2;
    // Spread each texel's albedo mass across the four nearest spawn tiles
    // instead of assigning it to one. Removes the tile discontinuity; costs a
    // wider scan of shared memory and nothing else.
    bool      tent = true;

    float     bias = 1e-3f;        // camera offset along its own normal, world units
    glm::vec3 sky{0.0f};           // radiance of an uncovered texel
    float     emissive = 1.0f;
    bool      two_sided = false;
    float     plane_tol = 0.05f;   // upsample plane cutoff, world units
    bool      running = true;

    // Only the fields that change buffer sizes.
    bool layout_equals(const SolveConfig& o) const {
        if (bounces != o.bounces || scale != o.scale || budget != o.budget) return false;
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

    Pipeline place_, raster_, gather_, upsample_, direct_px_, filter_;

    // Per level: cameras, irradiance, the direct term with its visible fraction,
    // and the per-tile masses that spawned them (two vec4s per child: the
    // indirect mass and the direct mass -- see raster.comp).
    std::array<gl::Buffer, kMaxLevels> cams_, irrad_, direct_, weights_;
    std::array<Quadrature, kMaxLevels> quad_;
    Quadrature direct_quad_;       // for the per-pixel direct pass
    std::array<LevelInfo, kMaxLevels> info_{};
    uint32_t levels_ = 0;

    gl::Texture gi_{gl::TextureType::tex_2d};      // GI grid, RGBA32F, (E, 1)
    gl::Texture gi_tmp_{gl::TextureType::tex_2d};  // ping-pong for the denoise
    gl::Texture full_{gl::TextureType::tex_2d};    // framebuffer resolution
    int gi_w_ = 0, gi_h_ = 0, full_w_ = 0, full_h_ = 0;

    SolveConfig layout_{};
    bool allocated_ = false;
    uint32_t cursor_ = 0, sweeps_ = 0, last_chunk_ = 0;
    uint32_t dump_mode_ = 1;

    PassTimer t_place_{"Place"};
    std::array<PassTimer, kMaxLevels> t_raster_{
        PassTimer{"Raster L1"}, PassTimer{"Raster L2"},
        PassTimer{"Raster L3"}, PassTimer{"Raster L4"}};
    PassTimer t_gather_{"Resolve"};
    PassTimer t_upsample_{"Upsample"};
    PassTimer t_direct_{"Direct/px"};
    PassTimer t_filter_{"GI filter"};
};

} // namespace mbg

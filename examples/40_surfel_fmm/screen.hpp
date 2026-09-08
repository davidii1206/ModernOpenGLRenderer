#pragma once

// ---------------------------------------------------------------------------
// The screen-space side: G-buffer, geometry pass, the surfel -> pixel gather,
// the debug point cloud, and the display/compare pass.
//
// The reconstruction is a per-pixel GATHER over the uniform grid: each pixel
// collects the surfels within a few spacings of it and fits them (surfel_gather
// .comp). A point-sprite SCATTER and a per-pixel microbuffer reprojection were
// both built and measured against it -- findings 12, 16 and 17 -- and both were
// removed once the gather won, so those findings' numbers can no longer be
// reproduced from this tree.
// ---------------------------------------------------------------------------

#include "gpu_util.hpp"
#include "surfels.hpp"
#include "grid.hpp"
#include "brute.hpp"
#include "cuts.hpp"

namespace sgi {

// Deferred G-buffer.
//
//   gbuf0  RGBA8   albedo.rgb | a: 1 = geometry, 0 = background
//   gbuf1  RGBA16F normal.xyz (world) | w: roughness
//   gbuf2  RGBA16F emissive.rgb | w: metallic
//   depth  DEPTH32F
//
// No motion vector target: nothing in this example is temporal.
struct GBuffer {
    gl::Framebuffer fbo;
    gl::Texture albedo, normal, emissive, depth;
    int width = 0, height = 0;

    void create(int w, int h);
    void bind_for_geometry() const;
    void bind_textures() const;      // units 0..3
};

// Draws every mesh of the model with its own node transform.
class GeometryPass {
public:
    bool init();
    bool poll() { return prog_.poll(); }
    void render(const GBuffer& gb, const gfx::Model& model, const glm::mat4& view_proj);
private:
    Pipeline prog_;
};

// Object-space denoise of the irradiance cache, for display only.
//
// The mottling that reads as "you can see the surfels" is noise in the cached E
// (7.4x the reference's high frequency on a provably smooth floor patch), not a
// reconstruction artifact. Blurring it away in the reconstruction tied denoising
// to sharpness. This unties them: filter here, reconstruct sharply after.
//
// The solve keeps running on the UNFILTERED buffer, so transport and every gate
// are untouched.
class SurfelFilterPass {
public:
    bool init();
    bool poll() { return prog_.poll(); }
    // `radius` and `plane_tol` in surfel spacings. Returns the buffer holding the
    // result, which the caller binds over kBindIrrad for the reconstruction.
    // `iterations` 0 returns the set's own irradiance untouched.
    // `version` identifies the irradiance the filter would be run on. The result
    // is cached against it, so a converged or paused solve pays nothing and the
    // iteration count stops being a per-frame cost -- which is what makes a
    // usefully strong denoise affordable at all.
    gl::Buffer* run(SurfelSet& set, const SurfelGrid& grid, int iterations,
                    float radius, float plane_tol, float normal_tol,
                    float light_sigma, float grad_scale, uint64_t version);
private:
    void ensure(uint32_t count);
    Pipeline prog_, grad_;
    gl::Buffer a_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer b_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer grad_buf_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    uint32_t count_ = 0;
    uint64_t cached_version_ = ~0ull;
    int cached_iters_ = -1;
    float cached_radius_ = -1.0f;
    float cached_sigma_ = -1.0f;
    float cached_grad_ = -1.0f;
    gl::Buffer* cached_result_ = nullptr;
};

// The irradiance cache, reconstructed as a per-pixel GATHER over the uniform
// grid: degree-1 moving least squares over the surfels within a few spacings,
// with plane, normal and light-visibility edge stops.
class SurfelGatherPass {
public:
    bool init();
    bool poll() { return prog_.poll(); }
    void resize(int w, int h);
    // `radius` and `plane_tol` are in surfel SPACINGS; the pass converts. The
    // spacing is the unit because the bake ties it to the radius exactly.
    void render(const GBuffer& gb, SurfelSet& set, const SurfelGrid& grid,
                const gfx::Camera& cam, float radius, float plane_tol,
                float normal_tol, int kernel, float light_sigma,
                float grad_scale, bool show_light, int mls, bool debug_fallback,
                gl::Buffer* irrad = nullptr);
    const gl::Texture& target() const { return accum_; }
    // [min, max] of light_vis over the surfels that fed each pixel. The direct
    // pass uses it to decide whether the march can be skipped; it cannot ride in
    // accum_.a, which display.frag divides by.
    const gl::Texture& bracket() const { return bracket_; }
private:
    Pipeline prog_;
    gl::Texture accum_{gl::TextureType::tex_2d};
    gl::Texture bracket_{gl::TextureType::tex_2d};
    int width_ = 0, height_ = 0;
};

// The direct term, per pixel.
//
// Runs the SAME estimator as the per-surfel direct pass (common/nee.glsl) with
// the receiver taken from the G-buffer, and ADDS the result into the
// reconstruction's target. The cache carries the direct term for the bounce
// transport either way, so this changes only what the image reads -- and it
// takes the sharpest term in the image off the surfel cache's Nyquist limit,
// which is the thing that no amount of interpolation could fix.
//
// Compositing this way needs a target that holds E directly, which the gather's
// is -- the scatter reconstruction's was a (w*E, w) accumulator and could not
// receive it.
class DirectPixelPass {
public:
    bool init();
    bool poll() { return prog_.poll(); }
    void render(const GBuffer& gb, SurfelSet& set, const SurfelGrid& grid,
                const EmitterSet& emitters, const CutSet& cuts,
                const gfx::Camera& cam,
                const gl::Texture& target, const gl::Texture& bracket,
                int width, int height,
                const SolveConfig& cfg, bool show_light);
private:
    Pipeline prog_;
};

// Raw, unsmoothed point cloud. The check that the gather above is not hiding
// something.
class SurfelPointsPass {
public:
    bool init();
    bool poll() { return prog_.poll(); }
    void render(const GBuffer& gb, SurfelSet& set, const glm::mat4& view_proj,
                const gfx::Camera& cam, int color_mode, float point_scale,
                float irradiance_gain);
    static const char* const* color_mode_names(int& count);
private:
    Pipeline prog_;
    GLuint empty_vao_ = 0;
};

// Path-traced references, loaded from PNG and compared against in display space.
struct References {
    gfx::Texture direct;    // CornellBoxGroundTruthDirectLighting.png
    gfx::Texture full;      // CornellBoxOriginalGroundTruth.png
    bool have_direct = false, have_full = false;

    // Loads both and disables sRGB decode, so texture() returns the stored
    // display-space bytes. The references are already tonemapped by the path
    // tracer, so the comparison has to happen after our own tonemap, not in
    // linear light where we would have to undo a curve we do not know.
    void load();
};

class DisplayPass {
public:
    struct Params {
        int   view_mode = 7;
        float exposure = 1.0f;
        float irradiance_gain = 1.0f;
        float diff_gain = 4.0f;
        float split_x = 0.5f;
        int   gt_index = 0;            // 0 = direct reference, 1 = full-GI reference
        int   tonemap = 0;             // see shaders/common/tonemap.glsl
        glm::mat4 inv_view_proj{1.0f};
        glm::vec3 scene_min{0.0f}, scene_extent{1.0f};
    };

    bool init();
    bool poll() { return prog_.poll(); }
    void render(const GBuffer& gb, const gl::Texture& recon,
                const References& refs, const Params& p);
    static const char* const* view_mode_names(int& count);

private:
    Pipeline prog_;
    GLuint empty_vao_ = 0;
    GLuint dummy_ = 0;
};

} // namespace sgi

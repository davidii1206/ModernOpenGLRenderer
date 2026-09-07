#pragma once

// ---------------------------------------------------------------------------
// S7/S8 screen-space side of the pipeline. At M0 this is the deferred G-buffer
// and the debug display pass; the screen gather, bilateral upsample, direct
// shading and the glossy ladder land here in M4/M5.
// ---------------------------------------------------------------------------

#include "gpu_util.hpp"
#include "mosaic.hpp"
#include "surfel_bake.hpp"

namespace mosaic {

// Deferred G-buffer. See shaders/common/gbuffer.glsl for the target layout.
//
// gl::Texture::image_2d allocates IMMUTABLE storage, so a resize cannot
// re-spec in place — glTextureStorage2D on a live texture is INVALID_OPERATION
// and silently keeps the old size. create() therefore move-assigns fresh
// textures every time.
struct GBuffer {
    gl::Framebuffer fbo;
    gl::Texture albedo, normal, emissive, motion, depth;
    int width = 0, height = 0;

    void create(int w, int h);
    void bind_for_geometry() const;
    // Binds albedo/normal/emissive/motion/depth to units 0..4.
    void bind_textures() const;
};

// Renders every instance into the G-buffer.
class GeometryPass {
public:
    bool init();
    bool poll() { return prog_.poll(); }

    void render(const GBuffer& gb,
                const std::vector<Instance>& instances,
                const std::vector<const gfx::Model*>& models,
                const glm::mat4& view_proj,
                const glm::mat4& prev_view_proj);

private:
    Pipeline prog_;
};

// Everything the lit composite needs. Passed as a struct because the list only
// grows as M5 adds emitter tiers.
//
// Namespace scope, not nested in DisplayPass: a nested class's default member
// initializers are not complete until the enclosing class is, so `const
// Lighting& = Lighting{}` as a default argument of a DisplayPass member is
// ill-formed. DisplayPass::Lighting stays valid through the alias below.
struct Lighting {
    const gl::Texture* indirect = nullptr;
    GLuint sun_shadow = 0;
    glm::mat4 sun_view_proj{1.0f};
    glm::vec3 sun_dir{0.0f, 1.0f, 0.0f};
    glm::vec3 sun_radiance{0.0f};
    float shadow_texel = 0.0f;
    bool have_sun = false;
    float emissive_boost = 1.0f;
    glm::vec3 sky{0.0f};
    glm::vec3 cam_pos{0.0f};
    class EmitterSet* emitters = nullptr;
    class Clipmap* clip = nullptr;      // supplies the occupancy field
    int emitter_steps = 16;
    float indirect_gain = 1.0f;
};

// Fullscreen tonemap / debug-view pass.
class DisplayPass {
public:
    using Lighting = mosaic::Lighting;

    bool init();
    bool poll() { return prog_.poll(); }

    void render(const GBuffer& gb, int view_mode, float exposure,
                const glm::mat4& inv_view_proj,
                const glm::vec3& scene_min, const glm::vec3& scene_extent,
                const Lighting& light = Lighting{});

    static const char* const* view_mode_names(int& count);

private:
    Pipeline prog_;
    GLuint empty_vao_ = 0;
    GLuint dummy_shadow_ = 0;
};

// S7 — half-res screen gather plus bilateral upsample. Turns the per-surfel
// cache into a full-resolution indirect-irradiance buffer.
class ScreenGI {
public:
    bool init();
    bool poll();

    void resize(int full_w, int full_h);

    // `sh_cache` is the filtered cache; `clip` supplies the cascade-0 index grid.
    void render(class Clipmap& clip, const GBuffer& gb, const gfx::Camera& cam,
                gl::Buffer& sh_cache, const glm::vec3& sky, int search_radius,
                float reach);

    const gl::Texture& indirect() const { return full_; }
    const gl::Texture& indirect_half() const { return half_; }

    enum SubPass { kGather, kUpsample, kSubPassCount };
    static const char* subpass_name(int i);
    PassTimer& subpass(int i) { return *sub_[i]; }
    double gpu_ms() const {
        double t = 0.0;
        for (int i = 0; i < kSubPassCount; ++i) t += sub_[i]->disp_gpu();
        return t;
    }

private:
    Pipeline gather_;
    Pipeline upsample_;
    gl::Texture half_{gl::TextureType::tex_2d};
    gl::Texture full_{gl::TextureType::tex_2d};
    int width_ = 0, height_ = 0;
    std::unique_ptr<PassTimer> sub_[kSubPassCount];
};

// Debug point-cloud view of the baked object-space surfel sets, drawn with the
// instance transform so it doubles as a preview of S0's object->world step.
class SurfelDebugPass {
public:
    bool init();
    bool poll() { return prog_.poll(); }

    void render(SurfelLibrary& lib,
                const std::vector<Instance>& instances,
                const glm::mat4& view_proj,
                const GBuffer& gb,
                // lod < 0 draws the LOD S0 actually selected for each instance
                // this frame, which is the only set that is live and therefore
                // the only one the cache ever updates.
                int lod, int color_mode, float point_scale,
                float near_clip, float far_clip,
                gl::Buffer* sh_cache = nullptr, float irradiance_gain = 1.0f,
                const std::vector<int>* live_lod = nullptr);

    static const char* const* color_mode_names(int& count);

private:
    Pipeline prog_;
    GLuint empty_vao_ = 0;
};

} // namespace mosaic

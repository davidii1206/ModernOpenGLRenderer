#pragma once

// ---------------------------------------------------------------------------
// S2, S3, S5 and S6 — the irradiance cache itself.
//
//   S2  direct lighting on every live surfel, every frame
//   S3  update scheduler: priority, 256-bin bucketing, compaction
//   S5  micro-render gather, one workgroup per selected surfel
//   S6  SH projection and the variance-adaptive temporal blend
//
// The cache is stored per SOURCE surfel, not per live surfel: S0 reassigns live
// indices every frame as LOD selection and instance ordering change, so a cache
// keyed on the live index would scramble whenever either did. That indirection
// is also what makes the object-space claim hold — a rigid instance's cached
// irradiance survives its motion untouched.
// ---------------------------------------------------------------------------

#include "clipmap.hpp"
#include "direct.hpp"
#include "gpu_util.hpp"
#include "mosaic.hpp"
#include "screen.hpp"
#include "surfel_bake.hpp"

namespace mosaic {

// Binds the occupancy cascades and their addressing for visibility.glsl.
// Shared by S2 and the pixel composite, which want the same field at different
// step counts.
void bind_occupancy(const Pipeline& p, Clipmap& clip, int first_unit);

struct SunLight {
    glm::vec3 direction{0.35f, 0.85f, 0.30f};   // points TOWARD the sun
    glm::vec3 color{1.0f, 0.96f, 0.88f};
    float intensity = 3.0f;
    bool enabled = false;
};

class CacheGI {
public:
    bool init(uint32_t source_surfel_count);
    bool poll();

    // S2: outgoing radiance per live surfel. Must run BEFORE the cluster
    // aggregation, which reads it.
    void direct_lighting(Clipmap& clip, const SunLight& sun,
                         const glm::mat4& sun_view_proj, GLuint sun_shadow,
                         float shadow_texel, float bounce_gain, float emissive_boost,
                         class EmitterSet& emitters, int emitter_steps);

    // S3 + S5 + S6.
    void update_cache(Clipmap& clip, const GBuffer& gb, const gfx::Camera& cam,
                      const Config& cfg, const glm::vec3& sky_radiance,
                      int shell_radius, int max_individual,
                      float temporal_min, float temporal_max, uint32_t frame);

    void reset_history() { reset_history_ = true; }

    // Feeds the gather a constant hemisphere so the SH round trip can be
    // checked numerically (plan gate 3).
    void set_debug_constant(bool on, float radiance) {
        debug_constant_ = on;
        debug_radiance_ = radiance;
    }

    // One-shot readback of the persistent cache: how much of it has ever been
    // written, and the distribution of the DC coefficient. Stalls; debug only.
    void log_stats(const SurfelLibrary* lib = nullptr) const;

    // The buffer consumers should read (filtered when the filter is enabled).
    gl::Buffer& sh_cache() { return spatial_strength_ > 0.0f ? sh_filtered_ : sh_cache_; }
    gl::Buffer& sh_raw() { return sh_cache_; }
    float& spatial_strength() { return spatial_strength_; }
    uint32_t selected_count() const { return selected_count_; }

    enum SubPass { kDirect, kSchedule, kGather, kSpatial, kSubPassCount };
    static const char* subpass_name(int i);
    PassTimer& subpass(int i) { return *sub_[i]; }
    double gpu_ms() const {
        double t = 0.0;
        for (int i = 0; i < kSubPassCount; ++i) t += sub_[i]->disp_gpu();
        return t;
    }

private:
    Pipeline direct_;
    Pipeline age_;
    Pipeline priority_;
    Pipeline cutoff_;
    Pipeline select_;
    Pipeline indirect_;
    Pipeline gather_;
    Pipeline spatial_;

    // Persistent, keyed by source surfel index: 4 x vec4 (see sh.glsl).
    // Raw cache: written by the gather, read by the spatial filter.
    gl::Buffer sh_cache_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    // Filtered cache: what every consumer reads. Kept separate so the blur never
    // feeds back into the temporal estimate.
    gl::Buffer sh_filtered_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer bucket_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer histogram_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer cutoff_buf_{gl::BufferType::shader, gl::BufferUsage::dynamic_read};
    gl::Buffer selected_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer cmd_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};

    std::unique_ptr<PassTimer> sub_[kSubPassCount];
    uint32_t source_count_ = 0;
    uint32_t selected_count_ = 0;
    uint32_t budget_ = 8192;
    bool reset_history_ = true;
    float spatial_strength_ = 0.75f;
    bool debug_constant_ = false;
    float debug_radiance_ = 1.0f;
    int stats_countdown_ = 0;

public:
    uint32_t& budget() { return budget_; }
};

} // namespace mosaic

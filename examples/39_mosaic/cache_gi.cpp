#include "cache_gi.hpp"

#include <gllib/log.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cstdio>
#include <cmath>

namespace mosaic {

const char* CacheGI::subpass_name(int i) {
    static const char* names[] = {"S2 direct", "S3 schedule", "S5 gather", "S6 spatial"};
    return names[i];
}

bool CacheGI::init(uint32_t source_surfel_count) {
    for (int i = 0; i < kSubPassCount; ++i)
        sub_[i] = std::make_unique<PassTimer>(subpass_name(i));

    direct_   = Pipeline::compute("shaders/surfel_direct.comp");
    age_      = Pipeline::compute("shaders/sh_age.comp");
    priority_ = Pipeline::compute("shaders/sched_priority.comp");
    cutoff_   = Pipeline::compute("shaders/sched_cutoff.comp");
    select_   = Pipeline::compute("shaders/sched_select.comp");
    indirect_ = Pipeline::compute("shaders/sched_indirect.comp");
    gather_   = Pipeline::compute("shaders/micro_gather.comp");
    spatial_  = Pipeline::compute("shaders/sh_spatial.comp");

    source_count_ = source_surfel_count;
    if (source_count_ == 0) return false;

    // Zero-initialised: a cold cache reads as black, and the gather's
    // age == 0 test then treats the first update as a history reset.
    std::vector<glm::vec4> zero(size_t(source_count_) * 4, glm::vec4(0.0f));
    sh_cache_.data(zero.data(), zero.size() * sizeof(glm::vec4));
    sh_filtered_.data(zero.data(), zero.size() * sizeof(glm::vec4));

    histogram_.data(nullptr, 256 * sizeof(uint32_t));
    cutoff_buf_.data(nullptr, 4 * sizeof(uint32_t));
    cmd_.data(nullptr, 3 * sizeof(uint32_t));

    return direct_.valid() && age_.valid() && priority_.valid() && cutoff_.valid() &&
           select_.valid() && indirect_.valid() && gather_.valid() && spatial_.valid();
}

bool CacheGI::poll() {
    bool c = direct_.poll();
    c |= age_.poll();
    c |= priority_.poll();
    c |= cutoff_.poll();
    c |= select_.poll();
    c |= indirect_.poll();
    c |= gather_.poll();
    c |= spatial_.poll();
    return c;
}

// Binds the occupancy cascades and their addressing for visibility.glsl.
// Shared by S2 and the pixel composite, which want the same field at different
// step counts.
void bind_occupancy(const Pipeline& p, Clipmap& clip, int first_unit) {
    const int n = clip.active_cascades();
    for (int l = 0; l < Config::kCascades; ++l) {
        char name[48];
        // Every element of a sampler array must be bound, even the ones the
        // loop never reads: sampling an unbound sampler3D is undefined.
        const int src = std::min(l, n - 1);
        glBindTextureUnit(GLuint(first_unit + l), clip.occupancy(src).handle());
        std::snprintf(name, sizeof(name), "u_occupancy[%d]", l);
        p.set(name, first_unit + l);
        std::snprintf(name, sizeof(name), "u_occ_origins[%d]", l);
        p.set(name, clip.cascade(src).origin);
        std::snprintf(name, sizeof(name), "u_occ_inv_cells[%d]", l);
        // Occupancy is kOccRes across the same volume the cluster cascade
        // covers, so its cell is finer by exactly that ratio.
        p.set(name, float(Config::kOccRes) / float(Config::kClipRes) /
                        clip.cascade(src).cell);
    }
    p.set("u_occ_cascades", n);
    p.set("u_occ_mips", clip.occupancy_mips());
}

void CacheGI::direct_lighting(Clipmap& clip, const SunLight& sun,
                              const glm::mat4& sun_view_proj, GLuint sun_shadow,
                              float shadow_texel, float bounce_gain,
                              float emissive_boost, EmitterSet& emitters,
                              int emitter_steps) {
    const uint32_t live = clip.live_count();
    if (live == 0) { sub_[kDirect]->skip(); return; }

    sub_[kDirect]->begin();
    clip.live_pn().bind_base(0);
    clip.live_alb().bind_base(1);
    clip.live_emissive().bind_base(2);
    clip.live_src().bind_base(3);
    sh_filtered_.bind_base(4);      // S2 uses the filtered result
    clip.live_radiance().bind_base(5);
    emitters.buffer().bind_base(6);
    emitters.proxied_buffer().bind_base(7);

    if (sun_shadow != 0) glBindTextureUnit(0, sun_shadow);
    bind_occupancy(direct_, clip, 1);

    direct_.use();
    direct_.set("u_live_count", live);
    direct_.set("u_sun_dir", glm::normalize(sun.direction));
    direct_.set("u_sun_radiance", sun.color * sun.intensity);
    direct_.set("u_sun_view_proj", sun_view_proj);
    direct_.set("u_shadow_texel", shadow_texel);
    direct_.set("u_bounce_gain", bounce_gain);
    direct_.set("u_emissive_boost", emissive_boost);
    direct_.set("u_have_sun", sun.enabled && sun_shadow != 0 ? 1 : 0);
    direct_.set("u_emitter_count", emitters.count());
    direct_.set("u_emitter_steps", emitter_steps);
    gl::dispatch_compute((live + 255) / 256, 1, 1);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    sub_[kDirect]->end();
}

void CacheGI::update_cache(Clipmap& clip, const GBuffer& gb, const gfx::Camera& cam,
                           const Config& cfg, const glm::vec3& sky_radiance,
                           int shell_radius, int max_individual,
                           float temporal_min, float temporal_max, uint32_t frame) {
    const uint32_t live = clip.live_count();
    if (live == 0) { sub_[kSchedule]->skip(); sub_[kGather]->skip(); return; }

    if (bucket_.size() < size_t(live) * sizeof(uint32_t))
        bucket_.data(nullptr, size_t(live) * sizeof(uint32_t));
    if (selected_.size() < size_t(budget_) * sizeof(uint32_t))
        selected_.data(nullptr, size_t(budget_) * sizeof(uint32_t));

    // --- S3 ---------------------------------------------------------------
    sub_[kSchedule]->begin();

    // Age every cached surfel first, so a surfel the gather refreshes below
    // ends this frame at age 0 rather than 1.
    sh_cache_.bind_base(0);
    age_.use();
    age_.set("u_surfel_count", source_count_);
    gl::dispatch_compute((source_count_ + 255) / 256, 1, 1);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    clear_uint_buffer(histogram_);
    clip.live_pn().bind_base(0);
    clip.live_src().bind_base(1);
    sh_cache_.bind_base(2);
    bucket_.bind_base(3);
    histogram_.bind_base(4);
    gb.depth.bind(0);

    priority_.use();
    priority_.set("u_live_count", live);
    priority_.set("u_view_proj", cam.view_projection());
    priority_.set("u_eye", cam.position());
    priority_.set("u_screen", glm::vec2(float(gb.width), float(gb.height)));
    priority_.set("u_near", cam.near_clip());
    priority_.set("u_far", cam.far_clip());
    priority_.set("u_frame", frame);
    // Pixels per radian: screen height over the vertical angular extent.
    priority_.set("u_px_scale",
                  float(gb.height) / (2.0f * std::tan(glm::radians(cam.fov() * 0.5f))));
    gl::dispatch_compute((live + 255) / 256, 1, 1);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    histogram_.bind_base(0);
    cutoff_buf_.bind_base(1);
    cutoff_.use();
    cutoff_.set("u_budget", budget_);
    gl::dispatch_compute(1, 1, 1);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    bucket_.bind_base(0);
    cutoff_buf_.bind_base(1);
    selected_.bind_base(2);
    select_.use();
    select_.set("u_live_count", live);
    select_.set("u_budget", budget_);
    select_.set("u_frame", frame);
    gl::dispatch_compute((live + 255) / 256, 1, 1);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    cutoff_buf_.bind_base(0);
    cmd_.bind_base(1);
    indirect_.use();
    indirect_.set("u_budget", budget_);
    gl::dispatch_compute(1, 1, 1);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);
    sub_[kSchedule]->end();

    // --- S5 + S6 ----------------------------------------------------------
    sub_[kGather]->begin();
    clip.live_pn().bind_base(0);
    clip.live_radiance().bind_base(1);
    clip.live_src().bind_base(2);
    clip.cell_sc().bind_base(3);
    clip.packed_idx().bind_base(4);
    clip.cluster_pn().bind_base(5);
    clip.cluster_rad().bind_base(6);
    clip.cell_mask().bind_base(7);
    selected_.bind_base(8);
    sh_cache_.bind_base(9);
    cutoff_buf_.bind_base(10);
    clip.far_cells().bind_base(11);

    gather_.use();
    for (int l = 0; l < clip.active_cascades(); ++l) {
        char name[64];
        const Cascade& c = clip.cascade(l);
        std::snprintf(name, sizeof(name), "u_origins[%d]", l);   gather_.set(name, c.origin);
        std::snprintf(name, sizeof(name), "u_inv_cells[%d]", l); gather_.set(name, 1.0f / c.cell);
        std::snprintf(name, sizeof(name), "u_cells[%d]", l);     gather_.set(name, c.cell);
        std::snprintf(name, sizeof(name), "u_cnt_offs[%d]", l);  gather_.set(name, c.cnt_off);
        std::snprintf(name, sizeof(name), "u_mask_offs[%d]", l); gather_.set(name, c.mask_off);
    }
    gather_.set("u_cascades", clip.active_cascades());
    gather_.set("u_shell_radius", shell_radius);
    gather_.set("u_max_far_cells", Clipmap::kMaxFarCells);
    gather_.set("u_max_individual", max_individual);
    gather_.set("u_sky_radiance", sky_radiance);
    gather_.set("u_temporal_min", temporal_min);
    gather_.set("u_temporal_max", temporal_max);
    gather_.set("u_frame", frame);
    gather_.set("u_reset_history", reset_history_ ? 1 : 0);
    gather_.set("u_debug_constant", debug_constant_ ? 1 : 0);
    gather_.set("u_debug_radiance", debug_radiance_);
    reset_history_ = false;

    // gl::BufferType has no dispatch_indirect, so bind the target raw.
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, cmd_.handle());
    gl::dispatch_compute_indirect(0);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    sub_[kGather]->end();

    // --- S6b spatial filter ---------------------------------------------
    sub_[kSpatial]->begin();
    clip.live_pn().bind_base(0);
    clip.live_src().bind_base(1);
    sh_cache_.bind_base(2);
    sh_filtered_.bind_base(3);
    clip.cell_sc().bind_base(4);
    clip.packed_idx().bind_base(5);
    spatial_.use();
    spatial_.set("u_origin", clip.cascade(0).origin);
    spatial_.set("u_inv_cell", 1.0f / clip.cascade(0).cell);
    spatial_.set("u_cnt_off", clip.cascade(0).cnt_off);
    spatial_.set("u_live_count", live);
    spatial_.set("u_strength", spatial_strength_);
    gl::dispatch_compute((live + 255) / 256, 1, 1);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    sub_[kSpatial]->end();

    // Selected count is UI only, so read it back occasionally rather than
    // stalling every frame for one integer.
    if (--stats_countdown_ <= 0) {
        stats_countdown_ = 30;
        uint32_t c[2] = {};
        glGetNamedBufferSubData(cutoff_buf_.handle(), 0, sizeof(c), c);
        selected_count_ = std::min(c[1], budget_);
    }
}

void CacheGI::log_stats(const SurfelLibrary* lib) const {
    if (source_count_ == 0) return;
    std::vector<glm::vec4> data(size_t(source_count_) * 4);
    glGetNamedBufferSubData(sh_cache_.handle(), 0,
                            GLsizeiptr(data.size() * sizeof(glm::vec4)), data.data());

    // Peak reconstructed irradiance, evaluated along the direction the L1 band
    // points in, which is the maximum of sh_eval_irradiance over all normals:
    //   E/pi = C0 * c0 + (2/3) * C1 * |c1|
    // (the basis constants are already folded into the stored coefficients, so
    // they appear once here, not squared)
    // For the constant-hemisphere test this must come back as exactly the input
    // radiance, independent of the surfel's orientation.
    constexpr double C0 = 0.282095, C1 = 0.488603, A1_OVER_A0 = 2.0 / 3.0;

    size_t nonzero = 0;
    double sum = 0.0, maxv = 0.0;
    double age_sum = 0.0, age_max = 0.0;
    double fresh = 0.0;   // updated within the last 32 frames
    for (uint32_t i = 0; i < source_count_; ++i) {
        const glm::vec3 c0 = glm::vec3(data[size_t(i) * 4 + 0]);
        const glm::vec3 cy = glm::vec3(data[size_t(i) * 4 + 1]);
        const glm::vec3 cz = glm::vec3(data[size_t(i) * 4 + 2]);
        const glm::vec3 cx = glm::vec3(data[size_t(i) * 4 + 3]);

        auto luma = [](const glm::vec3& v) {
            return 0.2126 * v.r + 0.7152 * v.g + 0.0722 * v.b;
        };
        const double l0 = luma(c0);
        const glm::dvec3 l1(luma(cx), luma(cy), luma(cz));
        const double peak = C0 * l0 + A1_OVER_A0 * C1 * glm::length(l1);

        if (std::abs(l0) > 1e-7) ++nonzero;
        sum += peak;
        maxv = std::max(maxv, peak);

        const double age = data[size_t(i) * 4 + 1].w;
        age_sum += age;
        age_max = std::max(age_max, age);
        if (age < 32.0) fresh += 1.0;
    }
    // Per-entry freshness: one library entry is one mesh's surfel set, so a
    // starved entry names the mesh directly.
    if (lib) {
        for (const auto& e : lib->entries()) {
            const uint32_t lo = e.base;
            const uint32_t hi = e.base + e.set.total();
            size_t fresh_b = 0;
            double age_b = 0.0;
            for (uint32_t i = lo; i < hi; ++i) {
                const double age = data[size_t(i) * 4 + 1].w;
                age_b += age;
                if (age < 32.0) ++fresh_b;
            }
            const uint32_t n = hi - lo;
            gllib::logf(gllib::LogLevel::info,
                        "  entry model %d mesh %d [%u,%u) n=%u: %.1f%% fresh, age mean %.1f",
                        e.model, e.mesh, lo, hi, n,
                        n ? 100.0 * double(fresh_b) / double(n) : 0.0,
                        n ? age_b / double(n) : 0.0);
        }
    }

    gllib::logf(gllib::LogLevel::info,
                "cache: %zu/%u written (%.1f%%), %.1f%% fresh (age<32); "
                "peak irradiance mean %.5f max %.5f; age mean %.1f max %.0f",
                nonzero, source_count_, 100.0 * double(nonzero) / double(source_count_),
                100.0 * fresh / double(source_count_),
                sum / double(source_count_), maxv,
                age_sum / double(source_count_), age_max);
}

} // namespace mosaic

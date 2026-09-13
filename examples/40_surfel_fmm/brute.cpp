#include "brute.hpp"

#include <gllib/log.hpp>

#include <algorithm>
#include <cmath>

namespace sgi {

// --- bucket table -----------------------------------------------------------

namespace {

glm::vec3 hemi_oct_decode(float ex, float ey) {
    const glm::vec2 t((ex + ey) * 0.5f, (ex - ey) * 0.5f);
    return glm::normalize(glm::vec3(t.x, t.y, 1.0f - std::abs(t.x) - std::abs(t.y)));
}

// Van Oosterom & Strackee: the solid angle of the spherical triangle (a,b,c).
// Numerically well behaved for the very thin triangles the octahedron's vertices
// produce, which the naive Girard-by-dihedral-angles form is not.
double tri_solid_angle(const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c) {
    const double num = std::abs(glm::dot(a, glm::cross(b, c)));
    const double den = 1.0 + glm::dot(a, b) + glm::dot(b, c) + glm::dot(c, a);
    return 2.0 * std::atan2(num, den);
}

} // namespace

std::vector<glm::vec4> build_bucket_table(uint32_t ms) {
    std::vector<glm::vec4> out(std::size_t(ms) * ms * 2);

    // dw exactly, from the four spherical corners of each texel.
    auto corner = [&](uint32_t i, uint32_t j) {
        const float ex = float(i) / float(ms) * 2.0f - 1.0f;
        const float ey = float(j) / float(ms) * 2.0f - 1.0f;
        return glm::dvec3(hemi_oct_decode(ex, ey));
    };

    // wcos and the mean direction by subsampling with the map's analytic
    // Jacobian. The square -> octahedron unfold is area-preserving and
    // octahedron -> sphere is a central projection, so dOmega goes as 1/|p|^3.
    const uint32_t S = 16;
    const double   du = 2.0 / double(ms * S);

    double wsum_all = 0.0;
    std::vector<double> sw(std::size_t(ms) * ms, 0.0);
    std::vector<glm::dvec3> sd(std::size_t(ms) * ms, glm::dvec3(0.0));
    std::vector<double> scos(std::size_t(ms) * ms, 0.0);

    for (uint32_t by = 0; by < ms; ++by) {
        for (uint32_t bx = 0; bx < ms; ++bx) {
            const std::size_t b = std::size_t(by) * ms + bx;
            for (uint32_t sy = 0; sy < S; ++sy) {
                for (uint32_t sx = 0; sx < S; ++sx) {
                    const double ex = (double(bx * S + sx) + 0.5) * du - 1.0;
                    const double ey = (double(by * S + sy) + 0.5) * du - 1.0;
                    const double tx = (ex + ey) * 0.5, ty = (ex - ey) * 0.5;
                    const glm::dvec3 p(tx, ty, 1.0 - std::abs(tx) - std::abs(ty));
                    const double len = glm::length(p);
                    const double w = 1.0 / (len * len * len);
                    const glm::dvec3 d = p / len;
                    sw[b] += w;
                    sd[b] += w * d;
                    scos[b] += w * d.z;
                    wsum_all += w;
                }
            }
        }
    }
    const double norm = 6.283185307179586 / wsum_all;   // make the square carry 2*PI

    double check_dw = 0.0, check_wcos = 0.0;
    for (uint32_t by = 0; by < ms; ++by) {
        for (uint32_t bx = 0; bx < ms; ++bx) {
            const std::size_t b = std::size_t(by) * ms + bx;
            const glm::dvec3 c00 = corner(bx, by),     c10 = corner(bx + 1, by);
            const glm::dvec3 c11 = corner(bx + 1, by + 1), c01 = corner(bx, by + 1);
            const double dw = tri_solid_angle(c00, c10, c11) + tri_solid_angle(c00, c11, c01);
            const glm::dvec3 dir = glm::normalize(sd[b]);
            const double wcos = scos[b] * norm;

            out[b * 2 + 0] = glm::vec4(float(dir.x), float(dir.y), float(dir.z), float(dw));
            out[b * 2 + 1] = glm::vec4(float(wcos), 0.0f, 0.0f, 0.0f);
            check_dw += dw;
            check_wcos += wcos;
        }
    }

    gllib::logf(gllib::LogLevel::info,
                "bucket table %ux%u: sum dw = %.7f (2pi = 6.2831853), "
                "sum wcos = %.7f (pi = 3.1415927)",
                ms, ms, check_dw, check_wcos);
    return out;
}

// --- Solver -----------------------------------------------------------------

bool Solver::init() {
    lout_prog_ = Pipeline::compute("shaders/bf_lout.comp");
    radiance_  = Pipeline::compute("shaders/bf_radiance.comp");
    micro_     = Pipeline::compute("shaders/bf_micro.comp");
    lod_prog_  = Pipeline::compute("shaders/lod_tier.comp");
    p2m_       = Pipeline::compute("shaders/fmm_p2m.comp");
    m2m_       = Pipeline::compute("shaders/fmm_m2m.comp");
    m2l_       = Pipeline::compute("shaders/fmm_m2l.comp");
    l2l_       = Pipeline::compute("shaders/fmm_l2l.comp");
    direct_    = Pipeline::compute("shaders/nee_direct.comp");
    blk_prog_  = Pipeline::compute("shaders/blk_rad.comp");
    timer_     = std::make_unique<PassTimer>("Solve");
    return lout_prog_.valid() && radiance_.valid() && micro_.valid() &&
           blk_prog_.valid() && lod_prog_.valid();
}

bool Solver::poll() {
    bool changed = lout_prog_.poll();
    changed |= lod_prog_.poll();
    changed |= p2m_.poll();
    changed |= m2m_.poll();
    changed |= m2l_.poll();
    changed |= l2l_.poll();
    changed |= radiance_.poll();
    changed |= micro_.poll();
    changed |= blk_prog_.poll();
    return changed;
}

void Solver::reset(SurfelSet& set) {
    set.reset_irradiance();
    cursor_ = 0;
    sweeps_ = 0;
    stats_ = SolveStats{};
    snapshot_.clear();
}

void Solver::ensure_buffers(const SurfelSet& set, uint32_t ms, bool tiered) {
    if (lout_count_ != set.count()) {
        const std::vector<glm::vec4> zero(set.count(), glm::vec4(0.0f));
        b_lout_.data(zero.data(), zero.size() * sizeof(glm::vec4));
        lout_count_ = set.count();
    }
    if (bucket_ms_ != ms || bucket_tiered_ != tiered) {
        // One table per LOD tier, end to end. Each is exact for its own
        // resolution -- dw and wcos come from that texel's spherical corners --
        // so a coarse tier cannot index into the fine table and read solid
        // angles that belong to a quarter of its own texel.
        const uint32_t edge[3] = {ms, std::max(4u, ms / 2u), std::max(4u, ms / 4u)};
        const uint32_t tiers = tiered ? 3u : 1u;
        std::vector<glm::vec4> tbl;
        for (uint32_t t = 0; t < 3; ++t) {
            bucket_edge_[t] = t < tiers ? edge[t] : ms;
            bucket_off_[t]  = t < tiers ? uint32_t(tbl.size()) : 0u;
            if (t >= tiers) continue;
            const std::vector<glm::vec4> one = build_bucket_table(edge[t]);
            tbl.insert(tbl.end(), one.begin(), one.end());
        }
        b_bucket_.data(tbl.data(), tbl.size() * sizeof(glm::vec4));
        bucket_ms_ = ms;
        bucket_tiered_ = tiered;
    }
}


void Solver::bucket_sums(uint32_t ms, double& sum_dw, double& sum_wcos) {
    const std::vector<glm::vec4> tbl = build_bucket_table(ms);
    sum_dw = 0.0; sum_wcos = 0.0;
    for (std::size_t b = 0; b * 2 + 1 < tbl.size(); ++b) {
        sum_dw += double(tbl[b * 2 + 0].w);
        sum_wcos += double(tbl[b * 2 + 1].x);
    }
}

bool Solver::nee_active(const SolveConfig& cfg) const {
    // The sun counts as an emitter here even though it is not IN the emitter
    // buffer: it is synthesized per receiver (see nee.glsl). A scene with no
    // emissive surface and a sun -- which is every outdoor scene -- would
    // otherwise take the NEE path's guard as "there is no direct light" and skip
    // the pass entirely.
    return cfg.nee && grid_ != nullptr && grid_->valid() &&
           ((emitters_ != nullptr && emitters_->count() > 0) || cfg.sun_is_nee_light());
}

// Passes 3-5 of the pipeline: the upsweep, the M2L, and the downsweep. Run once
// per sweep, immediately after lout, because the multipole is a function of
// outgoing radiance and nothing else.
//
// The whole tree is rebuilt every sweep rather than updated. Spec section 7 is
// emphatic about this: a partially updated local expansion is spatially
// inconsistent and reads as blotching, which is far worse than latency, and
// these three passes are cheap next to pass 6 by design.
void Solver::run_fmm(SurfelSet& set, const SolveConfig& cfg) {
    if (fmm_ == nullptr || !fmm_->valid() || grid_ == nullptr) return;
    if (!p2m_.valid() || !m2m_.valid() || !m2l_.valid() || !l2l_.valid()) return;

    set.bind();
    grid_->bind();
    fmm_->bind();
    b_lout_.bind_base(kBindLout);

    const uint32_t L = fmm_->levels();
    auto groups = [](uint32_t n) { return (n + 63u) / 64u; };
    auto regions = [&](const Pipeline& p) {
        p.set("u_fmm_cell_off", fmm_->cell_off());
        p.set("u_fmm_ilm_off", fmm_->ilm_off());
        p.set("u_fmm_lsh_off", fmm_->lsh_off());
    };

    // P2M at the leaves. This also clears Lsh at level 0; M2M clears it above.
    {
        const FmmTree::Level& lv = fmm_->level(0);
        p2m_.use();
        regions(p2m_);
        p2m_.set("u_slots", lv.slots);
        p2m_.set("u_slot_base", lv.slot_base);
        p2m_.set("u_coeff_base", lv.coeff_base);
        p2m_.set("u_two_sided", cfg.two_sided ? 1u : 0u);
        gl::dispatch_compute(groups(lv.slots), 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // M2M upward. Every level must be complete before the next reads it, so the
    // barrier is inside the loop and not after it.
    for (uint32_t l = 1; l < L; ++l) {
        const FmmTree::Level& lv = fmm_->level(l);
        const FmmTree::Level& ch = fmm_->level(l - 1);
        m2m_.use();
        regions(m2m_);
        m2m_.set("u_slots", lv.slots);
        m2m_.set("u_coeff_base", lv.coeff_base);
        m2m_.set("u_res", lv.res);
        m2m_.set("u_child_slot_base", ch.slot_base);
        m2m_.set("u_child_coeff_base", ch.coeff_base);
        m2m_.set("u_child_res", ch.res);
        gl::dispatch_compute(groups(lv.slots), 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // M2L at every level. Independent between levels -- each reads its own Ilm
    // and writes its own Lsh -- so one barrier at the end covers all of them.
    for (uint32_t l = 0; l < L; ++l) {
        const FmmTree::Level& lv = fmm_->level(l);
        m2l_.use();
        regions(m2l_);
        m2l_.set("u_slots", lv.slots);
        m2l_.set("u_slot_base", lv.slot_base);
        m2l_.set("u_coeff_base", lv.coeff_base);
        m2l_.set("u_res", lv.res);
        m2l_.set("u_h", grid_->cell() * float(1u << l));
        gl::dispatch_compute(groups(lv.slots), 1, 1);
    }
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // L2L downward, parent into child, so the leaf ends up carrying every level
    // of the expansion. Top-down and barriered per level for the same reason the
    // upsweep is.
    for (uint32_t l = L - 1; l-- > 0;) {
        const FmmTree::Level& lv = fmm_->level(l);
        const FmmTree::Level& pa = fmm_->level(l + 1);
        l2l_.use();
        regions(l2l_);
        l2l_.set("u_slots", lv.slots);
        l2l_.set("u_coeff_base", lv.coeff_base);
        l2l_.set("u_res", lv.res);
        l2l_.set("u_parent_slot_base", pa.slot_base);
        l2l_.set("u_parent_coeff_base", pa.coeff_base);
        l2l_.set("u_parent_res", pa.res);
        gl::dispatch_compute(groups(lv.slots), 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
}

void Solver::run_lout(SurfelSet& set, const SolveConfig& cfg) {
    set.bind();
    b_lout_.bind_base(kBindLout);
    if (direct_count_ != 0) b_direct_.bind_base(kBindDirect);
    const bool nee = nee_active(cfg);
    lout_prog_.use();
    lout_prog_.set("u_count", set.count());
    lout_prog_.set("u_emissive_scale", cfg.emissive_scale);
    lout_prog_.set("u_nee", nee ? 1u : 0u);
    // The cache holds the residual alone when the image takes its direct term per
    // pixel, so put it back here -- see bf_lout.comp.
    lout_prog_.set("u_add_direct", (nee && cfg.nee_pixel) ? 1u : 0u);
    gl::dispatch_compute((set.count() + 255u) / 256u, 1, 1);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // The order-0 multipole rides on lout, so it is rebuilt exactly when lout
    // is: once per sweep, one pass over the surfels. Two dispatches because a
    // clear and an accumulate cannot share a barrier.
    if (grid_ != nullptr && grid_->valid() && blk_prog_.valid()) {
        const glm::ivec3 mr = grid_->macro_res();
        const uint32_t words = uint32_t(mr.x) * uint32_t(mr.y) * uint32_t(mr.z) * 16u;
        if (words != blk_words_) {
            const std::vector<uint32_t> zero(words, 0u);
            b_blk_.data(zero.data(), zero.size() * sizeof(uint32_t));
            blk_words_ = words;
        }
        b_blk_.bind_base(kBindBlkRad);
        blk_prog_.use();
        blk_prog_.set("u_count", set.count());
        blk_prog_.set("u_words", words);
        blk_prog_.set("u_grid_min", grid_->min());
        blk_prog_.set("u_macro_res", mr);
        blk_prog_.set("u_macro_cell", grid_->cell() * 4.0f);
        blk_prog_.set("u_clear", 1u);
        gl::dispatch_compute((words + 255u) / 256u, 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
        blk_prog_.set("u_clear", 0u);
        gl::dispatch_compute((set.count() + 255u) / 256u, 1, 1);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    if (cfg.fmm) run_fmm(set, cfg);
}

// The direct term is a function of geometry and emission only -- it does not
// depend on the sweep's irradiance -- so it is computed once per solve and reused
// across every sweep. Recomputed only when something it actually depends on moves.
void Solver::run_direct(SurfelSet& set, const SolveConfig& cfg) {
    if (!cfg.nee) return;
    if (!direct_.valid() || grid_ == nullptr || emitters_ == nullptr) return;
    if (!grid_->valid()) return;
    if (emitters_->count() == 0 && !cfg.sun_is_nee_light()) return;

    if (direct_count_ != set.count()) {
        const std::vector<glm::vec4> zero(set.count(), glm::vec4(0.0f));
        b_direct_.data(zero.data(), zero.size() * sizeof(glm::vec4));
        direct_count_ = set.count();
        direct_dirty_ = true;
    }
    // This pass runs BEFORE dispatch(), which is where the visibility buffer used
    // to be allocated -- so on the first call the shader was writing light_vis
    // into a buffer with no storage and every value read back as zero, while the
    // direct term (allocated here) was correct. Allocate it here too.
    if (light_count_ != set.count()) {
        const std::vector<float> zero(set.count(), 0.0f);
        b_light_.data(zero.data(), zero.size() * sizeof(float));
        const uint32_t zmax = 0u;
        b_light_max_.data(&zmax, sizeof(uint32_t));
        light_count_ = set.count();
    }
    if (!direct_dirty_ && direct_thick_ == cfg.nee_thick &&
        direct_bias_ == cfg.nee_bias && direct_scale_ == cfg.emissive_scale &&
        direct_self_cos_ == cfg.nee_self_cos && direct_self_tol_ == cfg.nee_self_tol &&
        direct_occ_ == cfg.nee_occ && direct_cuts_ == int(cfg.nee_cuts))
        return;

    set.bind();
    grid_->bind();
    emitters_->bind();
    if (cuts_) cuts_->bind();
    b_direct_.bind_base(kBindDirect);
    b_light_.bind_base(kBindLightVis);

    direct_.use();
    direct_.set("u_count", set.count());
    direct_.set("u_emitters", emitters_->count());
    direct_.set("u_grid_min", grid_->min());
    direct_.set("u_grid_res", grid_->res());
    direct_.set("u_inv_cell", grid_->inv_cell());
    direct_.set("u_cell", grid_->cell());
    direct_.set("u_macro_res", grid_->macro_res());
    b_light_max_.bind_base(kBindLightMax);
    direct_.set("u_thick", cfg.nee_thick);
    direct_.set("u_bias", cfg.nee_bias);
    direct_.set("u_self_cos", cfg.nee_self_cos);
    direct_.set("u_sun_rad", cfg.sun_nee ? cfg.sun : glm::vec3(0.0f));
    direct_.set("u_sun_dir", glm::normalize(cfg.sun_dir));
    direct_.set("u_sun_dist", cfg.sun_dist);
    direct_.set("u_sun_half", cfg.sun_half);
    direct_.set("u_self_tol", cfg.nee_self_tol);
    direct_.set("u_occ", cfg.nee_occ);
    // No cut buffer attached (the analytic gates build their sets by hand and
    // have no source mesh) means no clipping, whatever the config says.
    direct_.set("u_cuts", (cfg.nee_cuts && cuts_ && cuts_->valid()) ? 1u : 0u);
    gl::dispatch_compute((set.count() + 63u) / 64u, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    direct_dirty_ = false;
    direct_thick_ = cfg.nee_thick;
    direct_bias_ = cfg.nee_bias;
    direct_scale_ = cfg.emissive_scale;
    direct_self_cos_ = cfg.nee_self_cos;
    direct_self_tol_ = cfg.nee_self_tol;
    direct_occ_ = cfg.nee_occ;
    direct_cuts_ = int(cfg.nee_cuts);
}

// Spec section 6.1. Sort this slice's receivers into three microbuffer
// resolutions by projected size and compact each tier into its own list.
//
// The workgroup counts are written by the shader's own atomics straight into an
// indirect command buffer. Reading them back to the host would stall the
// pipeline once per frame for three integers, which on this pass is a
// significant fraction of what the LOD is trying to save.
void Solver::classify_lod(SurfelSet& set, const SolveConfig& cfg,
                          uint32_t first, uint32_t slice) {
    if (lod_cap_ != slice) {
        const std::vector<uint32_t> zero(std::size_t(slice) * 3u, 0u);
        b_lod_list_.data(zero.data(), zero.size() * sizeof(uint32_t));
        lod_cap_ = slice;
    }
    // (count, 1, 1) three times. Only the counts are reset; y and z are written
    // every frame too because the whole command block is one upload either way.
    const uint32_t cmd[9] = {0u, 1u, 1u, 0u, 1u, 1u, 0u, 1u, 1u};
    b_lod_cmd_.data(cmd, sizeof(cmd));

    b_lod_list_.bind_base(kBindLodList);
    b_lod_cmd_.bind_base(kBindLodCmd);

    lod_prog_.use();
    lod_prog_.set("u_first", first);
    lod_prog_.set("u_slice", slice);
    lod_prog_.set("u_count", set.count());
    lod_prog_.set("u_cap", lod_cap_);
    lod_prog_.set("u_cam", cfg.cam_pos);
    lod_prog_.set("u_px_scale", cfg.px_scale);
    lod_prog_.set("u_ms", cfg.ms);
    lod_prog_.set("u_lod_px", cfg.lod_px);
    gl::dispatch_compute((slice + 255u) / 256u, 1, 1);
    // The command buffer is read by the dispatcher, not by a shader, so the
    // barrier has to name GL_COMMAND_BARRIER_BIT as well.
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

    // The tier split, once. A readback here is a full stall, so it happens on
    // the first classified slice only -- but without it the LOD is invisible:
    // an image that changes and a tier histogram nobody has seen is not a
    // measurement, and a rule that puts everything in tier 0 looks exactly like
    // a rule that works.
    if (!lod_logged_) {
        uint32_t cnt[9] = {};
        glGetNamedBufferSubData(b_lod_cmd_.handle(), 0, sizeof(cnt), cnt);
        gllib::logf(gllib::LogLevel::info,
                    "lod tiers (%u receivers): %ux%u %u, %ux%u %u, %ux%u %u",
                    slice, bucket_edge_[0], bucket_edge_[0], cnt[0],
                    bucket_edge_[1], bucket_edge_[1], cnt[3],
                    bucket_edge_[2], bucket_edge_[2], cnt[6]);
        lod_logged_ = true;
    }
}

void Solver::dispatch(SurfelSet& set, const SolveConfig& cfg,
                      uint32_t first, uint32_t slice, uint32_t frame) {
    if (slice == 0) return;
    set.bind();
    b_lout_.bind_base(kBindLout);
    b_bucket_.bind_base(kBindBucket);

    // A per-frame tangent frame (rotate 2) cannot be rebuilt by any reader that
    // knows only the surfel index -- it would silently reproject
    // through a different basis than the writer used. Refuse rather than produce
    // a plausible wrong image.
    // Emitter visibility is always written by the microbuffer solver: it is one
    // float per surfel and everything downstream that averages surfels together
    // needs it to avoid averaging across a shadow boundary.
    // The all-pairs microbuffer stores a GLOBAL surfel index in its 16-bit key
    // field, so above 65535 it would alias winners onto the wrong surfel and
    // report a plausible wrong image. The U-list path stores a local index and
    // has no such limit. Refuse rather than corrupt.
    // 65536, not 65535: a 16-bit field addresses 0..65535, so a set of exactly
    // 65536 is the largest one whose last index still fits. Gate 13's 32k rows
    // build precisely that, and an off-by-one here refused them.
    if (cfg.method == Method::Micro && set.count() > 65536u && cfg.near_radius <= 0.0f) {
        static bool said = false;
        if (!said) {
            gllib::logf(gllib::LogLevel::error,
                        "%u surfels with SGI_NEAR=0: the all-pairs microbuffer indexes "
                        "winners in 16 bits and cannot address this set. Set SGI_NEAR > 0 "
                        "to use the U-list path, or lower SGI_SURFELS.", set.count());
            said = true;
        }
        return;
    }

    if (cfg.method == Method::Micro && light_count_ != set.count()) {
        const std::vector<float> zero(set.count(), 0.0f);
        b_light_.data(zero.data(), zero.size() * sizeof(float));
        const uint32_t zmax = 0u;
        b_light_max_.data(&zmax, sizeof(uint32_t));
        light_count_ = set.count();
    }
    if (cfg.method == Method::Micro) {
        b_light_.bind_base(kBindLightVis);
        b_light_max_.bind_base(kBindLightMax);
    }

    const bool nee = nee_active(cfg);
    if (nee) b_direct_.bind_base(kBindDirect);

    if (cfg.method == Method::Radiance) {
        radiance_.use();
        radiance_.set("u_count", set.count());
        radiance_.set("u_first", first);
        radiance_.set("u_slice", slice);
        radiance_.set("u_horizon", cfg.horizon);
        radiance_.set("u_plane_bias", cfg.plane_bias);
        radiance_.set("u_self_cos", cfg.nee_self_cos);
        radiance_.set("u_soft_eps", cfg.soft_eps);
        radiance_.set("u_two_sided", cfg.two_sided ? 1u : 0u);
        // One THREAD per receiver; the candidate set is tiled through LDS.
        gl::dispatch_compute((slice + 255u) / 256u, 1, 1);
    } else {
        micro_.use();
        micro_.set("u_count", set.count());
        micro_.set("u_first", first);
        micro_.set("u_slice", slice);
        micro_.set("u_ms", cfg.ms);
        micro_.set("u_horizon", cfg.horizon);
        micro_.set("u_plane_bias", cfg.plane_bias);
        micro_.set("u_self_cos", cfg.nee_self_cos);
        micro_.set("u_near", cfg.near_radius);
        // The far-field occluder. The macro bitmask is a low-resolution binary
        // voxelization of the scene and it is already built; marching it per
        // bucket is what stops the far field arriving through walls.
        if (grid_ != nullptr && grid_->valid()) {
            grid_->bind();
            micro_.set("u_grid_min", grid_->min());
            micro_.set("u_macro_res", grid_->macro_res());
            micro_.set("u_macro_cell", grid_->cell() * 4.0f);
            micro_.set("u_grid_res", grid_->res());
            micro_.set("u_cell", grid_->cell());
            micro_.set("u_radius", set.radius());
            micro_.set("u_far_occ", cfg.far_occlusion ? 1u : 0u);
            micro_.set("u_far_order", uint32_t(cfg.far_order));
        const bool fmm = cfg.fmm && fmm_ != nullptr && fmm_->valid();
        micro_.set("u_fmm", fmm ? 1u : 0u);
        micro_.set("u_fmm_interp", cfg.fmm_interp ? 1u : 0u);
        micro_.set("u_fmm_slot_base", fmm ? fmm_->level(0).slot_base : 0u);
        micro_.set("u_fmm_coeff_base", fmm ? fmm_->level(0).coeff_base : 0u);
        if (fmm) {
            fmm_->bind();
            micro_.set("u_fmm_cell_off", fmm_->cell_off());
            micro_.set("u_fmm_ilm_off", fmm_->ilm_off());
            micro_.set("u_fmm_lsh_off", fmm_->lsh_off());
        }
            b_blk_.bind_base(kBindBlkRad);
        } else {
            micro_.set("u_far_occ", 0u);
        }
        micro_.set("u_soft_eps", cfg.soft_eps);
        micro_.set("u_two_sided", cfg.two_sided ? 1u : 0u);
        // The microbuffer spans the whole scene: there is no grid yet, so no h
        // to derive the spec's 2*sqrt(3)*h near-field depth from.
        const float max_depth = std::max(1e-3f, set.bounds().diagonal());
        micro_.set("u_depth_scale", 65535.0f / max_depth);
        micro_.set("u_depth_tol", cfg.depth_tol_radii * set.radius());
        micro_.set("u_normal_tol", cfg.normal_tol);
        micro_.set("u_sky", cfg.sky);
        micro_.set("u_sky_ground", cfg.sky_ground);
        micro_.set("u_sun", cfg.sun_nee ? glm::vec3(0.0f) : cfg.sun);
        micro_.set("u_sun_dir", glm::normalize(cfg.sun_dir));
        micro_.set("u_sun_cos", cfg.sun_cos);
        micro_.set("u_store_light", 1u);
        micro_.set("u_nee", nee ? 1u : 0u);
        micro_.set("u_nee_add", cfg.nee_pixel ? 0u : 1u);
        micro_.set("u_no_occlusion", cfg.no_occlusion ? 1u : 0u);
        micro_.set("u_rotate", uint32_t(std::max(0, cfg.rotate)));
        micro_.set("u_frame", frame);
        micro_.set("u_debug_constant", cfg.debug_constant ? 1u : 0u);
        micro_.set("u_debug_radiance", cfg.debug_radiance);

        // px_scale is 0 until a camera exists, and the gates run before one does
        // (they build synthetic scenes with no view at all). Zero would divide
        // into an infinite projected size and put every receiver in the top
        // tier, which is silently the same image at a higher cost -- so refuse
        // rather than pretend the LOD ran.
        const bool lod = cfg.lod && cfg.px_scale > 0.0f;
        if (cfg.lod && !lod) {
            static bool said = false;
            if (!said) {
                gllib::logf(gllib::LogLevel::warn,
                            "LOD asked for with no camera (px_scale = 0); "
                            "solving every receiver at %ux%u", cfg.ms, cfg.ms);
                said = true;
            }
        }

        if (!lod) {
            micro_.set("u_lod", 0u);
            micro_.set("u_lod_base", 0u);
            micro_.set("u_bucket_base", 0u);
            // One WORKGROUP per receiver: the microbuffer has to live in LDS.
            gl::dispatch_compute(slice, 1, 1);
        } else {
            classify_lod(set, cfg, first, slice);
            // Three specialized dispatches, not one kernel branching on tier:
            // section 6.1 is explicit that the branch costs more than the
            // dispatches. The counts come from the compaction's own atomics
            // through the indirect buffer, so nothing is read back.
            glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, b_lod_cmd_.handle());
            micro_.use();
            micro_.set("u_lod", 1u);
            for (uint32_t t = 0; t < 3; ++t) {
                micro_.set("u_ms", bucket_edge_[t]);
                micro_.set("u_bucket_base", bucket_off_[t]);
                micro_.set("u_lod_base", t * lod_cap_);
                gl::dispatch_compute_indirect(GLintptr(t * 3u * sizeof(uint32_t)));
            }
            glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
        }
    }

    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void Solver::end_sweep(SurfelSet& set, bool collect_stats) {
    // This sweep's output becomes the next one's input, then every receiver is
    // seeded with it so a partially advanced sweep never shows the display
    // two-sweep-old values -- the flicker section 7 describes, which section 7
    // itself sanctions fixing by "double-buffer but copy through".
    set.swap_irradiance();
    set.carry_irradiance();
    ++sweeps_;

    if (!collect_stats) { stats_.valid = false; return; }

    const std::vector<glm::vec4> cur = set.read_irradiance();
    const double disc_area = 3.14159265358979 * double(set.radius()) * double(set.radius());
    double sum = 0.0, peak = 0.0, delta = 0.0, flux = 0.0;
    uint32_t nonzero = 0;
    const bool have_prev = snapshot_.size() == cur.size();
    for (std::size_t i = 0; i < cur.size(); ++i) {
        const double l = 0.2126 * cur[i].r + 0.7152 * cur[i].g + 0.0722 * cur[i].b;
        sum += l;
        flux += l * disc_area;
        peak = std::max(peak, l);
        if (l > 1e-9) ++nonzero;
        if (have_prev) {
            const double p = 0.2126 * snapshot_[i].r + 0.7152 * snapshot_[i].g
                           + 0.0722 * snapshot_[i].b;
            delta += std::abs(l - p);
        }
    }
    const double n = double(std::max<std::size_t>(cur.size(), 1));
    stats_.mean = sum / n;
    stats_.peak = peak;
    stats_.flux = flux;
    stats_.delta = have_prev ? delta / n : 0.0;
    stats_.rel_delta = stats_.mean > 1e-12 ? stats_.delta / stats_.mean : 0.0;
    stats_.nonzero = nonzero;
    stats_.valid = true;
    snapshot_ = cur;
}

void Solver::step(SurfelSet& set, const SolveConfig& cfg, uint32_t frame, bool collect_stats) {
    if (set.count() == 0 || !cfg.running || converged(cfg)) { timer_->skip(); return; }
    ensure_buffers(set, cfg.ms, cfg.lod);

    const uint32_t n = set.count();
    const uint32_t budget = std::max(1u, std::min(cfg.budget, n));
    const uint32_t slice = std::min(budget, n - cursor_);

    {
        ScopedPass sp(*timer_);
        // L_out depends only on the previous sweep's irradiance, so it is
        // rebuilt once per sweep rather than once per slice.
        if (cursor_ == 0) { run_direct(set, cfg); run_lout(set, cfg); }
        dispatch(set, cfg, cursor_, slice, frame);
    }

    cursor_ += slice;
    if (cursor_ >= n) {
        cursor_ = 0;
        end_sweep(set, collect_stats);
    }
}

void Solver::run_sweeps(SurfelSet& set, const SolveConfig& cfg, uint32_t n, uint32_t frame) {
    if (set.count() == 0 || n == 0) return;
    ensure_buffers(set, cfg.ms, cfg.lod);

    for (uint32_t s = 0; s < n; ++s) {
        if (cursor_ != 0) {
            // Finish the slice already in flight first, so pressing "Solve now"
            // mid-sweep does not run part of the set twice.
            dispatch(set, cfg, cursor_, set.count() - cursor_, frame);
            cursor_ = 0;
            end_sweep(set, false);
            continue;
        }
        run_direct(set, cfg);
        run_lout(set, cfg);
        dispatch(set, cfg, 0, set.count(), frame + s);
        end_sweep(set, s + 1 == n);

        // Drain between sweeps. A full sweep is ~0.5 s at N = 30k and 16x16
        // buckets, and queueing a whole scripted solve without ever letting the
        // GPU go idle crosses the kernel's ~10 s job watchdog: the GPU is reset,
        // the context is lost, and every later frame comes back as flat grey
        // rather than as any kind of error. It reproduced exactly at the
        // boundary -- 20 sweeps fine, 24 dead. Finishing here bounds a single
        // submission to one sweep and costs nothing that matters, since this
        // path is the deliberately-blocking one.
        glFinish();
    }
}

} // namespace sgi

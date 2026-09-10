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
    direct_    = Pipeline::compute("shaders/nee_direct.comp");
    blk_prog_  = Pipeline::compute("shaders/blk_rad.comp");
    timer_     = std::make_unique<PassTimer>("Solve");
    return lout_prog_.valid() && radiance_.valid() && micro_.valid() &&
           blk_prog_.valid();
}

bool Solver::poll() {
    bool changed = lout_prog_.poll();
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

void Solver::ensure_buffers(const SurfelSet& set, uint32_t ms) {
    if (lout_count_ != set.count()) {
        const std::vector<glm::vec4> zero(set.count(), glm::vec4(0.0f));
        b_lout_.data(zero.data(), zero.size() * sizeof(glm::vec4));
        lout_count_ = set.count();
    }
    if (bucket_ms_ != ms) {
        const std::vector<glm::vec4> tbl = build_bucket_table(ms);
        b_bucket_.data(tbl.data(), tbl.size() * sizeof(glm::vec4));
        bucket_ms_ = ms;
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
    return cfg.nee && emitters_ != nullptr && emitters_->count() > 0 &&
           grid_ != nullptr && grid_->valid();
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
        const uint32_t words = uint32_t(mr.x) * uint32_t(mr.y) * uint32_t(mr.z) * 4u;
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
}

// The direct term is a function of geometry and emission only -- it does not
// depend on the sweep's irradiance -- so it is computed once per solve and reused
// across every sweep. Recomputed only when something it actually depends on moves.
void Solver::run_direct(SurfelSet& set, const SolveConfig& cfg) {
    if (!cfg.nee) return;
    if (!direct_.valid() || grid_ == nullptr || emitters_ == nullptr) return;
    if (emitters_->count() == 0 || !grid_->valid()) return;

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
            micro_.set("u_far_occ", cfg.far_occlusion ? 1u : 0u);
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
        micro_.set("u_store_light", 1u);
        micro_.set("u_nee", nee ? 1u : 0u);
        micro_.set("u_nee_add", cfg.nee_pixel ? 0u : 1u);
        micro_.set("u_no_occlusion", cfg.no_occlusion ? 1u : 0u);
        micro_.set("u_rotate", uint32_t(std::max(0, cfg.rotate)));
        micro_.set("u_frame", frame);
        micro_.set("u_debug_constant", cfg.debug_constant ? 1u : 0u);
        micro_.set("u_debug_radiance", cfg.debug_radiance);
        // One WORKGROUP per receiver: the microbuffer has to live in LDS.
        gl::dispatch_compute(slice, 1, 1);
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
    ensure_buffers(set, cfg.ms);

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
    ensure_buffers(set, cfg.ms);

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

#include "solver.hpp"

#include <gllib/log.hpp>

#include <algorithm>
#include <cmath>

namespace mbg {

namespace {

// A GL dispatch is capped per dimension, and the deep levels here run to
// millions of workgroups, so every dispatch is folded into a 2D grid. The
// shaders undo it with the same expression: flat = y * NumWorkGroups.x + x.
constexpr uint32_t kMaxGroupsX = 32768u;

void dispatch_groups(uint32_t groups) {
    if (groups == 0) return;
    const uint32_t x = std::min(groups, kMaxGroupsX);
    const uint32_t y = (groups + x - 1) / x;
    gl::dispatch_compute(x, y, 1);
}

// Largest divisor of `res` that is <= `block`. A tile edge that does not divide
// the target would silently drop the remainder column.
uint32_t snap_block(uint32_t res, uint32_t block) {
    block = std::clamp(block, 1u, res);
    while (res % block != 0u) --block;
    return block;
}

// Camera buffers are the whole memory cost of variant A, and the growth is
// geometric in the tile count. Rather than let a slider allocate 8 GB, the
// budget is clamped and the clamp is reported.
constexpr std::size_t kMaxCameraBytes = 512ull * 1024ull * 1024ull;
// Per camera: the camera itself, its irradiance, and TWO direct records -- the
// total with the emitters' visible fraction, and the sun's share with the sun's.
// Per child: three masses (indirect, emitter-direct, sun-direct). See
// raster.comp: the two lights' visibility varies at completely different rates
// and cannot share one scale factor.
constexpr std::size_t kCamBytes = 32, kIrradBytes = 16, kDirectBytes = 32,
                      kWeightBytes = 48;

} // namespace

bool Solver::init() {
    place_    = Pipeline::compute("shaders/place.comp");
    raster_   = Pipeline::compute("shaders/raster.comp");
    gather_   = Pipeline::compute("shaders/gather.comp");
    upsample_ = Pipeline::compute("shaders/upsample.comp");
    direct_px_ = Pipeline::compute("shaders/direct_pixel.comp");
    filter_ = Pipeline::compute("shaders/gi_filter.comp");
    return place_.valid() && raster_.valid() && gather_.valid() && upsample_.valid() &&
           direct_px_.valid() && filter_.valid();
}

bool Solver::poll() {
    bool changed = place_.poll();
    changed |= raster_.poll();
    changed |= gather_.poll();
    changed |= upsample_.poll();
    changed |= direct_px_.poll();
    changed |= filter_.poll();
    return changed;
}

// Children this level's cameras spawn.
//
// The whole difference between the two estimators lives in this function. Tile
// mode branches: every camera spawns one child per tile of its own target, so
// the count multiplies at each level and the tree is K^D. Path mode splits ONCE
// -- level 0 is the primary hit -- and every level below it continues each path
// with exactly one child, so the tree is a bundle of `paths` straight lines and
// its size is paths x D.
static uint32_t level_children(const SolveConfig& cfg, uint32_t l) {
    if (l + 1 >= cfg.bounces) return 0;                       // terminal
    if (cfg.paths) return l == 0 ? std::max(1u, cfg.paths) : 1u;
    const uint32_t bpr = cfg.res[l] / cfg.block[l];
    return bpr * bpr;
}

void Solver::configure(const SolveConfig& cfg_in, int fb_w, int fb_h) {
    SolveConfig cfg = cfg_in;
    cfg.bounces = std::clamp(cfg.bounces, 1u, kMaxLevels);
    cfg.scale   = std::clamp(cfg.scale, 1u, 64u);
    cfg.paths   = cfg.paths ? std::clamp(cfg.paths, 1u, 256u) : 0u;
    for (uint32_t l = 0; l < cfg.bounces; ++l) {
        cfg.res[l]   = std::clamp(cfg.res[l], 2u, 32u);
        cfg.block[l] = snap_block(cfg.res[l], cfg.block[l]);
    }

    const int gw = (fb_w + int(cfg.scale) - 1) / int(cfg.scale);
    const int gh = (fb_h + int(cfg.scale) - 1) / int(cfg.scale);
    if (allocated_ && layout_.layout_equals(cfg) && gw == gi_w_ && gh == gi_h_ &&
        fb_w == full_w_ && fb_h == full_h_)
        return;

    levels_ = cfg.bounces;
    gi_w_ = gw;
    gi_h_ = gh;

    // Level 0 never needs more cameras than the grid has pixels.
    uint32_t n = std::min(std::max(1u, cfg.budget), uint32_t(gi_w_ * gi_h_));

    // Size the tree once to find the clamp, then allocate.
    for (int attempt = 0; attempt < 32; ++attempt) {
        std::size_t total = 0;
        uint32_t k = n;
        for (uint32_t l = 0; l < levels_; ++l) {
            const uint32_t children = level_children(cfg, l);
            total += std::size_t(k) * (kCamBytes + kIrradBytes + kDirectBytes +
                                       (l ? kWeightBytes : 0));
            if (!children) break;
            if (k > 0 && children > (0xFFFFFFFFu / k)) { k = 0; break; }   // overflow guard
            k *= children;
        }
        if (total <= kMaxCameraBytes && k != 0) break;
        n = std::max(1u, n / 2);
        if (attempt == 31) break;
    }
    if (n != std::min(std::max(1u, cfg.budget), uint32_t(gi_w_ * gi_h_)))
        gllib::logf(gllib::LogLevel::warn,
                    "budget clamped to %u level-1 cameras to stay under %zu MB",
                    n, kMaxCameraBytes / (1024 * 1024));

    uint32_t count = n;
    for (uint32_t l = 0; l < levels_; ++l) {
        LevelInfo& li = info_[l];
        li.res = cfg.res[l];
        li.block = (l + 1 == levels_) ? 0u : cfg.block[l];
        li.children = level_children(cfg, l);
        li.cameras = count;
        li.bytes = std::size_t(count) * (kCamBytes + kIrradBytes + kDirectBytes +
                                        (l ? kWeightBytes : 0));

        const std::vector<uint8_t> zero(std::size_t(count) * kCamBytes, 0);
        cams_[l].data(zero.data(), zero.size());
        irrad_[l].data(nullptr, std::size_t(count) * kIrradBytes);
        direct_[l].data(nullptr, std::size_t(count) * kDirectBytes);
        if (l) weights_[l].data(nullptr, std::size_t(count) * kWeightBytes);

        quad_[l].build(li.res);
        count *= li.children;
        if (li.children == 0) break;
    }
    for (uint32_t l = levels_; l < kMaxLevels; ++l) info_[l] = LevelInfo{};

    auto make_image = [](gl::Texture& dst, int w, int h) {
        gl::Texture t(gl::TextureType::tex_2d);
        t.image_2d(0, GL_RGBA32F, w, h, GL_RGBA, GL_FLOAT, nullptr);
        t.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        t.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        t.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        t.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        dst = std::move(t);
    };
    // Both are rebuilt unconditionally: configure() only reaches this point when
    // something about the layout changed, and gl::Texture allocates IMMUTABLE
    // storage, so re-specing an existing one is INVALID_OPERATION and would
    // silently keep the old size.
    make_image(gi_, gi_w_, gi_h_);
    make_image(gi_tmp_, gi_w_, gi_h_);
    make_image(gi_disp_, gi_w_, gi_h_);
    make_image(full_, fb_w, fb_h);
    full_w_ = fb_w;
    full_h_ = fb_h;

    layout_ = cfg;
    allocated_ = true;
    cursor_ = 0;
    sweeps_ = 0;
    clear_image();
}

void Solver::clear_image() {
    const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (gi_.handle()) glClearTexImage(gi_.handle(), 0, GL_RGBA, GL_FLOAT, zero);
    if (gi_tmp_.handle()) glClearTexImage(gi_tmp_.handle(), 0, GL_RGBA, GL_FLOAT, zero);
    if (gi_disp_.handle()) glClearTexImage(gi_disp_.handle(), 0, GL_RGBA, GL_FLOAT, zero);
    if (full_.handle()) glClearTexImage(full_.handle(), 0, GL_RGBA, GL_FLOAT, zero);
}

void Solver::dispatch_1d(uint32_t count) {
    dispatch_groups((count + 63u) / 64u);
}

void Solver::step(const GBuffer& gb, const gfx::Camera& cam, const Scene& scene,
                  const SolveConfig& cfg) {
    if (!allocated_ || !place_.valid() || !raster_.valid() || !gather_.valid()) return;
    if (!cfg.running || gi_pixels() == 0) {
        t_place_.skip();
        for (uint32_t l = 0; l < kMaxLevels; ++l) t_raster_[l].skip();
        t_gather_.skip();
        // last_chunk_ deliberately keeps its value: a paused solver should still
        // report the size of the work it was doing, not zero.
        return;
    }

    const uint32_t chunk = std::min(info_[0].cameras, gi_pixels() - cursor_);
    last_chunk_ = chunk;
    const glm::mat4 inv_vp = glm::inverse(cam.view_projection());

    // --- 1. Place level-1 cameras on the GI grid ----------------------------
    {
        ScopedPass p(t_place_);
        place_.use();
        gb.bind_textures();
        cams_[0].bind_base(kBindCams);
        place_.set("u_cam_count", chunk);
        place_.set("u_cursor", cursor_);
        place_.set("u_gi_size", glm::ivec2(gi_w_, gi_h_));
        place_.set("u_size", glm::ivec2(gb.width, gb.height));
        place_.set("u_scale", layout_.scale);
        place_.set("u_inv_view_proj", inv_vp);
        dispatch_1d(chunk);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    run_levels(chunk, scene, cfg, true);

    cursor_ += chunk;
    if (cursor_ >= gi_pixels()) {
        cursor_ = 0;
        ++sweeps_;
    }
}

void Solver::raster_level(uint32_t l, uint32_t count, const Scene& scene,
                          const SolveConfig& cfg, gl::Buffer* dump) {
    const LevelInfo& li = info_[l];
    // 1.5x the scene diagonal: a camera in a corner looking at the far corner is
    // already at 1.0, and the packed key quantizes over this range.
    const float far = std::max(1e-4f, scene.bounds().diagonal() * 1.5f);

    raster_.use();
    scene.bind();
    cams_[l].bind_base(kBindCams);
    quad_[l].bind();
    irrad_[l].bind_base(kBindIrrad);
    direct_[l].bind_base(kBindDirect);
    if (li.children) {
        cams_[l + 1].bind_base(kBindChildCam);
        weights_[l + 1].bind_base(kBindChildW);
    } else {
        // Terminal level: the kernel writes neither, but leaving a binding point
        // stale invites a later edit to write into the wrong buffer.
        cams_[l].bind_base(kBindChildCam);
        irrad_[l].bind_base(kBindChildW);
    }
    if (dump) dump->bind_base(kBindVisOut);
    raster_.set("u_cam_count", count);
    raster_.set("u_tri_count", scene.count());
    raster_.set("u_cluster_count", scene.cluster_count());
    raster_.set("u_group_count", scene.group_count());
    // Only where there is something to amortize. Carrying four texels through
    // one traversal costs register pressure and a four-wide inner loop, and on
    // a scene that fits in cache that is a 1.5x LOSS -- measured on Cornell.
    raster_.set("u_coop",
                (cfg.coop && scene.cluster_count() > 8u) ? 1u : 0u);
    raster_.set("u_cull", cfg.cull ? 1u : 0u);
    // The order only matters if there are cluster levels to reorder.
    raster_.set("u_order", (cfg.order && cfg.cull) ? 1u : 0u);
    raster_.set("u_res", li.res);
    raster_.set("u_block", li.block);
    raster_.set("u_children", li.children);
    raster_.set("u_paths", cfg.paths ? 1u : 0u);
    raster_.set("u_level", l);
    // Roulette is a property of the path estimator; the tile estimator has no
    // single path to terminate, only a tile's worth of them at once.
    raster_.set("u_rr", cfg.paths ? std::max(0.0f, cfg.rr) : 0.0f);
    raster_.set("u_importance", cfg.importance ? 1u : 0u);
    raster_.set("u_bias", cfg.bias);
    raster_.set("u_inv_far", 1.0f / far);
    cfg.sky.bind(raster_);
    raster_.set("u_emissive", cfg.emissive);
    raster_.set("u_two_sided", cfg.two_sided ? 1u : 0u);
    raster_.set("u_dump", dump ? dump_mode_ : 0u);
    raster_.set("u_emitters", cfg.nee ? scene.emitter_count() : 0u);
    raster_.set("u_tent", cfg.tent ? 1u : 0u);
    raster_.set("u_lv_res", std::clamp(cfg.cam_lv_res, 2u, 32u));
    raster_.set("u_jitter", cfg.jitter ? 1u : 0u);
    dispatch_groups(count);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void Solver::run_levels(uint32_t chunk, const Scene& scene, const SolveConfig& cfg,
                        bool write_image) {
    // The GI image carries the residual whenever the direct term is not coming
    // from it -- because a per-pixel pass will add it back, or because
    // indirect_only wants it left out altogether.
    // The sun counts as an analytic direct term whether or not the emitter list
    // does: it has no geometry, so there is no quadrature path it could take
    // instead, and MBG_NEE only decides how the EMITTERS are found.
    const bool analytic = (cfg.nee && scene.emitter_count() > 0) || cfg.sky.has_sun();
    const bool split = write_image && analytic &&
                       (cfg.direct_pixel || cfg.indirect_only);
    // Rasterize shallowest first: a level's cameras are spawned by the level
    // above it, so level l+1's camera buffer is written by level l's raster.
    // Zero-initialized, then level 0 seeded: writing the trailing zeros out
    // would have to be re-counted every time kMaxLevels moves.
    uint32_t counts[kMaxLevels] = {};
    counts[0] = chunk;
    for (uint32_t l = 0; l < levels_; ++l) {
        ScopedPass p(t_raster_[l]);
        raster_level(l, counts[l], scene, cfg, nullptr);
        if (l + 1 < levels_) counts[l + 1] = counts[l] * info_[l].children;
    }
    for (uint32_t l = levels_; l < kMaxLevels; ++l) t_raster_[l].skip();

    // Resolve deepest first: E_parent needs its children finished.
    ScopedPass p(t_gather_);
    auto run = [&](uint32_t l, uint32_t children, bool write) {
        gather_.use();
        irrad_[l].bind_base(kBindIrrad);
        weights_[children ? l + 1 : l].bind_base(kBindChildW);
        irrad_[children ? l + 1 : l].bind_base(kBindChildE);
        direct_[children ? l + 1 : l].bind_base(kBindChildD);
        gi_.bind_image(0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        gather_.set("u_cam_count", counts[l]);
        gather_.set("u_children", children);
        gather_.set("u_write", write ? 1u : 0u);
        gather_.set("u_cursor", cursor_);
        gather_.set("u_gi_size", glm::ivec2(gi_w_, gi_h_));
        gather_.set("u_split", (write && split) ? 1u : 0u);
        direct_[l].bind_base(kBindDirect);
        dispatch_1d(counts[l]);
        gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    };
    if (levels_ == 1) {
        // One level: nothing to fold in, but the scatter still has to run.
        run(0, 0, write_image);
    } else {
        for (int l = int(levels_) - 2; l >= 0; --l)
            run(uint32_t(l), info_[uint32_t(l)].children, write_image && l == 0);
    }
}

void Solver::upload_points(const std::vector<glm::vec4>& pos,
                           const std::vector<glm::vec4>& nrm) {
    std::vector<glm::vec4> packed(pos.size() * 2);
    for (std::size_t i = 0; i < pos.size(); ++i) {
        packed[i * 2 + 0] = glm::vec4(glm::vec3(pos[i]), 1.0f);
        packed[i * 2 + 1] = glm::vec4(glm::vec3(nrm[i]), 1.0f);   // throughput
    }
    cams_[0].data(packed.data(), packed.size() * sizeof(glm::vec4));
}

std::vector<glm::vec4> Solver::solve_points(const Scene& scene, const SolveConfig& cfg,
                                            const std::vector<glm::vec4>& pos,
                                            const std::vector<glm::vec4>& nrm) {
    const uint32_t n = uint32_t(pos.size());
    if (n == 0) return {};
    SolveConfig c = cfg;
    c.scale = 1;
    c.budget = n;
    configure(c, int(n), 1);
    upload_points(pos, nrm);
    run_levels(n, scene, c, false);

    std::vector<glm::vec4> out(n);
    glGetNamedBufferSubData(irrad_[0].handle(), 0, GLsizeiptr(n * sizeof(glm::vec4)),
                            out.data());
    return out;
}

std::vector<uint32_t> Solver::raster_visibility(const Scene& scene, const SolveConfig& cfg,
                                                const std::vector<glm::vec4>& pos,
                                                const std::vector<glm::vec4>& nrm,
                                                uint32_t mode) {
    const uint32_t n = uint32_t(pos.size());
    if (n == 0) return {};
    SolveConfig c = cfg;
    c.scale = 1;
    c.budget = n;
    configure(c, int(n), 1);
    upload_points(pos, nrm);

    const uint32_t texels = info_[0].res * info_[0].res;
    gl::Buffer dump(gl::BufferType::shader, gl::BufferUsage::dynamic_read);
    dump.data(nullptr, std::size_t(n) * texels * sizeof(uint32_t));
    dump_mode_ = mode;
    raster_level(0, n, scene, c, &dump);
    dump_mode_ = 1;

    std::vector<uint32_t> out(std::size_t(n) * texels);
    glGetNamedBufferSubData(dump.handle(), 0,
                            GLsizeiptr(out.size() * sizeof(uint32_t)), out.data());
    return out;
}

void Solver::direct_pixel(const GBuffer& gb, const gfx::Camera& cam, const Scene& scene,
                          const SolveConfig& cfg) {
    const bool analytic = (cfg.nee && scene.emitter_count() > 0) || cfg.sky.has_sun();
    if (!allocated_ || !direct_px_.valid() || !cfg.direct_pixel ||
        cfg.indirect_only || !analytic) {
        t_direct_.skip();
        return;
    }
    ScopedPass p(t_direct_);
    // Its own quadrature table: this pass's hemisphere need not be the same size
    // as the GI grid's, and the table is what carries the texel weights the
    // visibility ratio is built from.
    // The light view is its own projection with its own solid-angle weights
    // computed in the shader, so it needs no quadrature table -- but the
    // hemisphere table stays bound because scene.glsl declares quad[] for every
    // pass that includes it.
    const uint32_t res = std::clamp(cfg.direct_res, 2u, 32u);
    if (direct_quad_.res != res) direct_quad_.build(res);
    direct_px_.use();
    scene.bind();
    direct_quad_.bind();
    gb.bind_textures();
    full_.bind_image(1, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    direct_px_.set("u_size", glm::ivec2(gb.width, gb.height));
    direct_px_.set("u_inv_view_proj", glm::inverse(cam.view_projection()));
    direct_px_.set("u_tri_count", scene.count());
    direct_px_.set("u_cluster_count", scene.cluster_count());
    direct_px_.set("u_cull", cfg.cull ? 1u : 0u);
    direct_px_.set("u_emitters", cfg.nee ? scene.emitter_count() : 0u);
    cfg.sky.bind(direct_px_);
    direct_px_.set("u_lv_res", res);
    direct_px_.set("u_inv_far", 1.0f / std::max(1e-4f, scene.bounds().diagonal() * 1.5f));
    direct_px_.set("u_two_sided", cfg.two_sided ? 1u : 0u);
    direct_px_.set("u_bias", cfg.bias);
    direct_px_.set("u_emissive", cfg.emissive);
    // One workgroup per pixel: this pass rasterizes a hemisphere per receiver,
    // exactly as the secondary cameras do.
    dispatch_groups(uint32_t(gb.width) * uint32_t(gb.height));
    gl::memory_barrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void Solver::filter(const GBuffer& gb, const gfx::Camera& cam, const SolveConfig& cfg) {
    if (!allocated_ || !filter_.valid() || cfg.filter_iters == 0) { t_filter_.skip(); return; }
    ScopedPass p(t_filter_);
    filter_.use();
    gb.bind_textures();
    filter_.set("u_gi_size", glm::ivec2(gi_w_, gi_h_));
    filter_.set("u_size", glm::ivec2(gb.width, gb.height));
    filter_.set("u_scale", layout_.scale);
    filter_.set("u_inv_view_proj", glm::inverse(cam.view_projection()));
    filter_.set("u_radius", std::max(1, cfg.filter_radius));
    filter_.set("u_plane_tol", std::max(1e-5f, cfg.plane_tol));

    // A-trous: the tap spacing doubles each iteration, so three passes of a 5x5
    // kernel reach as far as 17x17 for a ninth of the taps.
    //
    // gi_ IS READ ONCE, BY THE FIRST ITERATION, AND NEVER WRITTEN. The grid is an
    // accumulation: a sweep is split into chunks, so most of it was written
    // several frames ago and is not touched again until the cursor comes round.
    // Filtering in place therefore does not run the configured number of
    // iterations on a cell, it runs that many EVERY FRAME the cell sits there --
    // and an a-trous chain applied without bound converges to its own fixed
    // point, which is whatever the dilated passes have unity gain on. That is
    // the grid's Nyquist, period two cells, and on the ceiling it showed as
    // horizontal streaks that grew with the iteration count instead of shrinking
    // (implementation.md, finding 17).
    //
    // The ping-pong is arranged so the LAST iteration lands in gi_disp_, which
    // is what upsample() reads; no copy-back, and gi_ keeps the raw solve.
    const uint32_t n = cfg.filter_iters;
    for (uint32_t i = 0; i < n; ++i) {
        const gl::Texture& src = i == 0 ? gi_ : (((n - i) % 2 == 0) ? gi_disp_ : gi_tmp_);
        gl::Texture& dst = ((n - 1 - i) % 2 == 0) ? gi_disp_ : gi_tmp_;
        src.bind_image(0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
        dst.bind_image(1, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        filter_.set("u_stride", int(1u << i));
        gl::dispatch_compute(uint32_t((gi_w_ + 7) / 8), uint32_t((gi_h_ + 7) / 8), 1);
        gl::memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }
}

void Solver::upsample(const GBuffer& gb, const gfx::Camera& cam, const SolveConfig& cfg) {
    if (!allocated_ || !upsample_.valid()) { t_upsample_.skip(); return; }
    ScopedPass p(t_upsample_);
    upsample_.use();
    gb.bind_textures();
    // The denoised copy when there is one, the raw grid otherwise.
    (cfg.filter_iters > 0 ? gi_disp_ : gi_).bind_image(0, 0, GL_FALSE, 0,
                                                      GL_READ_ONLY, GL_RGBA32F);
    full_.bind_image(1, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    upsample_.set("u_size", glm::ivec2(gb.width, gb.height));
    upsample_.set("u_gi_size", glm::ivec2(gi_w_, gi_h_));
    upsample_.set("u_scale", layout_.scale);
    upsample_.set("u_inv_view_proj", glm::inverse(cam.view_projection()));
    upsample_.set("u_plane_tol", std::max(1e-5f, cfg.plane_tol));
    gl::dispatch_compute(uint32_t((gb.width + 7) / 8), uint32_t((gb.height + 7) / 8), 1);
    gl::memory_barrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

std::size_t Solver::bytes() const {
    std::size_t t = 0;
    for (uint32_t l = 0; l < levels_; ++l) t += info_[l].bytes;
    return t;
}

double Solver::chunk_cameras() const {
    double total = 0.0, k = double(last_chunk_);
    for (uint32_t l = 0; l < levels_; ++l) {
        total += k;
        k *= double(info_[l].children);
    }
    return total;
}

double Solver::sweep_cameras() const {
    double total = 0.0, k = double(gi_pixels());
    for (uint32_t l = 0; l < levels_; ++l) {
        total += k;
        k *= double(info_[l].children);
    }
    return total;
}

double Solver::sweep_texels() const {
    double total = 0.0, k = double(gi_pixels());
    for (uint32_t l = 0; l < levels_; ++l) {
        total += k * double(info_[l].res) * double(info_[l].res);
        k *= double(info_[l].children);
    }
    return total;
}

double Solver::chunk_texels() const {
    double total = 0.0, k = double(last_chunk_);
    for (uint32_t l = 0; l < levels_; ++l) {
        total += k * double(info_[l].res) * double(info_[l].res);
        k *= double(info_[l].children);
    }
    return total;
}

} // namespace mbg

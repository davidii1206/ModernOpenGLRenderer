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
constexpr std::size_t kCamBytes = 32, kIrradBytes = 16, kWeightBytes = 16;

} // namespace

bool Solver::init() {
    place_    = Pipeline::compute("shaders/place.comp");
    raster_   = Pipeline::compute("shaders/raster.comp");
    gather_   = Pipeline::compute("shaders/gather.comp");
    upsample_ = Pipeline::compute("shaders/upsample.comp");
    return place_.valid() && raster_.valid() && gather_.valid() && upsample_.valid();
}

bool Solver::poll() {
    bool changed = place_.poll();
    changed |= raster_.poll();
    changed |= gather_.poll();
    changed |= upsample_.poll();
    return changed;
}

void Solver::configure(const SolveConfig& cfg_in, int fb_w, int fb_h) {
    SolveConfig cfg = cfg_in;
    cfg.bounces = std::clamp(cfg.bounces, 1u, kMaxLevels);
    cfg.scale   = std::clamp(cfg.scale, 1u, 64u);
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
            const uint32_t block = (l + 1 == levels_) ? 0u : cfg.block[l];
            const uint32_t children = block ? (cfg.res[l] / block) * (cfg.res[l] / block) : 0u;
            total += std::size_t(k) * (kCamBytes + kIrradBytes + (l ? kWeightBytes : 0));
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
        li.children = li.block ? (li.res / li.block) * (li.res / li.block) : 0u;
        li.cameras = count;
        li.bytes = std::size_t(count) * (kCamBytes + kIrradBytes + (l ? kWeightBytes : 0));

        const std::vector<uint8_t> zero(std::size_t(count) * kCamBytes, 0);
        cams_[l].data(zero.data(), zero.size());
        irrad_[l].data(nullptr, std::size_t(count) * kIrradBytes);
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
    raster_.set("u_res", li.res);
    raster_.set("u_block", li.block);
    raster_.set("u_bias", cfg.bias);
    raster_.set("u_inv_far", 1.0f / far);
    raster_.set("u_sky", cfg.sky);
    raster_.set("u_emissive", cfg.emissive);
    raster_.set("u_two_sided", cfg.two_sided ? 1u : 0u);
    raster_.set("u_dump", dump ? 1u : 0u);
    dispatch_groups(count);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void Solver::run_levels(uint32_t chunk, const Scene& scene, const SolveConfig& cfg,
                        bool write_image) {
    // Rasterize shallowest first: a level's cameras are spawned by the level
    // above it, so level l+1's camera buffer is written by level l's raster.
    uint32_t counts[kMaxLevels] = {chunk, 0, 0, 0};
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
        gi_.bind_image(0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        gather_.set("u_cam_count", counts[l]);
        gather_.set("u_children", children);
        gather_.set("u_write", write ? 1u : 0u);
        gather_.set("u_cursor", cursor_);
        gather_.set("u_gi_size", glm::ivec2(gi_w_, gi_h_));
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
        packed[i * 2 + 1] = glm::vec4(glm::vec3(nrm[i]), 0.0f);
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
                                                const std::vector<glm::vec4>& nrm) {
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
    raster_level(0, n, scene, c, &dump);

    std::vector<uint32_t> out(std::size_t(n) * texels);
    glGetNamedBufferSubData(dump.handle(), 0,
                            GLsizeiptr(out.size() * sizeof(uint32_t)), out.data());
    return out;
}

void Solver::upsample(const GBuffer& gb, const gfx::Camera& cam, const SolveConfig& cfg) {
    if (!allocated_ || !upsample_.valid()) { t_upsample_.skip(); return; }
    ScopedPass p(t_upsample_);
    upsample_.use();
    gb.bind_textures();
    gi_.bind_image(0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
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

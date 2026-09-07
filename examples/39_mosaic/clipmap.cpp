#include "clipmap.hpp"

#include <gllib/log.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>

namespace mosaic {
namespace {

// Occupancy cascades cover the same world volume as the cluster cascades, at a
// higher resolution (kOccRes vs kClipRes), since a visibility field needs finer
// spatial detail than a radiance aggregate.
constexpr float kOccScale = float(Config::kOccRes) / float(Config::kClipRes);

void set_cascade_uniforms(const Pipeline& p, const Cascade* cascades) {
    // GLSL array uniforms are addressed element by element; there is no
    // uniform3fv-array setter in gl::Program.
    for (int l = 0; l < Config::kCascades; ++l) {
        char name[64];
        std::snprintf(name, sizeof(name), "u_origins[%d]", l);
        p.set(name, cascades[l].origin);
        std::snprintf(name, sizeof(name), "u_inv_cells[%d]", l);
        p.set(name, 1.0f / cascades[l].cell);
        std::snprintf(name, sizeof(name), "u_cnt_offs[%d]", l);
        p.set(name, cascades[l].cnt_off);
        std::snprintf(name, sizeof(name), "u_mask_offs[%d]", l);
        p.set(name, cascades[l].mask_off);
    }
}

} // namespace

const char* Clipmap::subpass_name(int i) {
    static const char* names[] = {"count", "scan", "compact", "merge", "cluster", "occupancy"};
    return names[i];
}

bool Clipmap::init() {
    for (int i = 0; i < kSubPassCount; ++i)
        sub_[i] = std::make_unique<PassTimer>(subpass_name(i));

    assemble_        = Pipeline::compute("shaders/assemble.comp");
    count_           = Pipeline::compute("shaders/clip_count.comp");
    bases_           = Pipeline::compute("shaders/clip_bases.comp");
    compact_         = Pipeline::compute("shaders/clip_compact.comp");
    merge_           = Pipeline::compute("shaders/clip_merge.comp");
    celllist_        = Pipeline::compute("shaders/clip_celllist.comp");
    cluster_cmd_     = Pipeline::compute("shaders/clip_cluster_cmd.comp");
    cluster_         = Pipeline::compute("shaders/clip_cluster.comp");
    occupancy_build_ = Pipeline::compute("shaders/occupancy.comp");
    occupancy_mip_   = Pipeline::compute("shaders/occupancy_mip.comp");
    if (!scan_.init()) return false;

    // Cell tables. Segments are laid out back to back, one per cascade, so a
    // single clear zeroes all of them.
    const uint32_t total_cells = uint32_t(Config::kClipCells) * Config::kCascades;
    const uint32_t total_mask = uint32_t((Config::kClipCells + 31) / 32) * Config::kCascades;
    cell_count_.data(nullptr, size_t(total_cells) * sizeof(uint32_t));
    cell_start_.data(nullptr, size_t(total_cells) * sizeof(uint32_t));
    cell_sc_.data(nullptr, size_t(total_cells) * 2 * sizeof(uint32_t));
    cell_mask_.data(nullptr, size_t(total_mask) * sizeof(uint32_t));
    cluster_pn_.data(nullptr, size_t(total_cells) * 2 * sizeof(glm::vec4));
    cluster_rad_.data(nullptr, size_t(total_cells) * sizeof(glm::vec4));

    for (int l = 0; l < Config::kCascades; ++l) {
        cascades_[l].cnt_off = uint32_t(l) * uint32_t(Config::kClipCells);
        cascades_[l].mask_off = uint32_t(l) * uint32_t((Config::kClipCells + 31) / 32);
    }

    // Occupancy: one mipped R8 3D texture per cascade. gl::Texture has no 3D
    // storage helper, so allocate through the DSA entry point directly.
    occ_mips_ = 1;
    while ((Config::kOccRes >> occ_mips_) >= 1) ++occ_mips_;
    for (int l = 0; l < Config::kCascades; ++l) {
        glTextureStorage3D(occupancy_[l].handle(), occ_mips_, GL_R8,
                           Config::kOccRes, Config::kOccRes, Config::kOccRes);
        occupancy_[l].parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        occupancy_[l].parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        occupancy_[l].parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        occupancy_[l].parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        occupancy_[l].parameter(GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    }

    stats_.data(nullptr, Config::kCascades * sizeof(uint32_t));
    far_cells_.data(nullptr, (kMaxFarCells + 1) * sizeof(uint32_t));
    cell_list_.data(nullptr, (kMaxListCells + 1) * sizeof(uint32_t));
    cluster_cmd_buf_.data(nullptr, 3 * sizeof(uint32_t));
    cascade_base_.data(nullptr, Config::kCascades * sizeof(uint32_t));

    return assemble_.valid() && count_.valid() && compact_.valid() && bases_.valid() &&
           merge_.valid() && cluster_.valid() && celllist_.valid() &&
           cluster_cmd_.valid() && occupancy_build_.valid() &&
           occupancy_mip_.valid();
}

bool Clipmap::poll() {
    bool changed = assemble_.poll();
    changed |= count_.poll();
    changed |= bases_.poll();
    changed |= compact_.poll();
    changed |= merge_.poll();
    changed |= celllist_.poll();
    changed |= cluster_cmd_.poll();
    changed |= cluster_.poll();
    changed |= occupancy_build_.poll();
    changed |= occupancy_mip_.poll();
    changed |= scan_.poll();
    return changed;
}

void Clipmap::resize_buffers(uint32_t capacity) {
    if (capacity <= live_capacity_) return;
    // Grow geometrically: the live count varies with LOD selection every frame,
    // and reallocating on each small increase would thrash.
    live_capacity_ = std::max(capacity, live_capacity_ + live_capacity_ / 2);

    live_pn_.data(nullptr, size_t(live_capacity_) * 2 * sizeof(glm::vec4));
    live_alb_.data(nullptr, size_t(live_capacity_) * sizeof(glm::vec4));
    live_emissive_.data(nullptr, size_t(live_capacity_) * sizeof(glm::vec4));
    live_meta_.data(nullptr, size_t(live_capacity_) * sizeof(uint32_t));
    live_src_.data(nullptr, size_t(live_capacity_) * sizeof(uint32_t));
    live_radiance_.data(nullptr, size_t(live_capacity_) * sizeof(glm::vec4));
    // One slot per surfel PER CASCADE, and packed_idx holds up to one entry per
    // surfel per cascade, now that a surfel is inserted into every cascade whose
    // box contains it rather than only the finest.
    surfel_slot_.data(nullptr, size_t(live_capacity_) * Config::kCascades * sizeof(uint32_t));
    packed_idx_.data(nullptr, size_t(live_capacity_) * Config::kCascades * sizeof(uint32_t));
}

uint32_t Clipmap::assemble(SurfelLibrary& lib,
                           const std::vector<Instance>& instances,
                           const gfx::Camera& cam,
                           const Config& cfg,
                           bool freeze_lod) {
    cpu_instances_.clear();
    instance_lod_.assign(instances.size(), -1);
    live_count_ = 0;
    budget_dropped_ = 0;

    if (lib.total_surfels() == 0) return 0;

    const glm::vec3 eye = cam.position();
    // Projected-size LOD selection.
    //
    // A world-space spacing h at distance d subtends h / (d * 2 tan(fov/2)) of
    // the screen height. Inverting that for a target on-screen spacing gives the
    // world spacing this instance actually needs; pick the coarsest LOD that is
    // still at least as fine as that. Working in screen fractions rather than in
    // absolute distance keeps the choice scene-scale agnostic.
    const float proj = 2.0f * std::tan(glm::radians(cam.fov() * 0.5f));
    constexpr float kTargetScreenSpacing = 1.0f / 220.0f;   // ~4 px at 900p

    uint32_t dst = 0;
    for (size_t i = 0; i < instances.size(); ++i) {
        const Instance& inst = instances[i];
        const int ei = lib.entry_for(inst);
        if (ei < 0) continue;
        const SurfelLibrary::Entry& e = lib.entries()[size_t(ei)];

        // Distance from the eye to the instance's world AABB centre.
        const glm::vec3 centre_obj = 0.5f * (e.set.aabb_min + e.set.aabb_max);
        const glm::vec3 centre_ws = glm::vec3(inst.xform * glm::vec4(centre_obj, 1.0f));
        const float dist = glm::length(centre_ws - eye);

        int lod = 0;
        if (!freeze_lod) {
            const float wanted = kTargetScreenSpacing * std::max(dist, 1e-4f) * proj;
            for (int l = Config::kSurfelLods - 1; l >= 0; --l) {
                if (cfg.surfel_spacing * Config::kLodSpacing[l] <= wanted) { lod = l; break; }
                lod = 0;
            }
            lod = std::clamp(lod, 0, Config::kSurfelLods - 1);
        }

        // Budget exhaustion: coarsen rather than drop. Falling back to a coarser
        // LOD keeps the instance in the cache at reduced density, which the
        // gather can still use; dropping it outright leaves a hole that reads as
        // missing indirect light. Only when even the coarsest level does not fit
        // is the instance skipped.
        while (lod < Config::kSurfelLods - 1 &&
               dst + e.set.lods[lod].count > Config::kMaxLiveSurfels)
            ++lod;

        const SurfelSet::Lod& L = e.set.lods[lod];
        if (L.count == 0) continue;
        if (dst + L.count > Config::kMaxLiveSurfels) {
            ++budget_dropped_;
            continue;
        }

        GpuInstance gi{};
        gi.xform = inst.xform;
        gi.aabb_min = glm::vec4(e.set.aabb_min, e.set.radius_scale);
        gi.aabb_ext = glm::vec4(e.set.aabb_max - e.set.aabb_min, e.object_scale);
        gi.range = glm::uvec4(e.base + L.offset, dst, L.count, inst.flags);
        cpu_instances_.push_back(gi);
        instance_lod_[i] = lod;
        dst += L.count;
    }

    live_count_ = dst;
    if (live_count_ == 0 || cpu_instances_.empty()) return 0;

    if (log_instances_) {
        log_instances_ = false;
        for (size_t k = 0; k < cpu_instances_.size(); ++k) {
            const GpuInstance& g = cpu_instances_[k];
            gllib::logf(gllib::LogLevel::info,
                        "  inst %2zu: src_base=%u dst_base=%u count=%u flags=%u",
                        k, g.range.x, g.range.y, g.range.z, g.range.w);
        }
    }

    resize_buffers(live_count_);
    instances_.data(cpu_instances_.data(), cpu_instances_.size() * sizeof(GpuInstance));

    lib.packed_buffer().bind_base(0);
    lib.emissive_dense_buffer().bind_base(1);
    instances_.bind_base(2);
    live_pn_.bind_base(3);
    live_alb_.bind_base(4);
    live_emissive_.bind_base(5);
    live_meta_.bind_base(6);
    live_src_.bind_base(7);

    assemble_.use();
    assemble_.set("u_live_count", live_count_);
    assemble_.set("u_instance_count", uint32_t(cpu_instances_.size()));
    gl::dispatch_compute((live_count_ + 255) / 256, 1, 1);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    return live_count_;
}

void Clipmap::build(const gfx::Camera& cam, const Config& cfg) {
    if (live_count_ == 0) return;

    // Camera-centred cascades, snapped to the cell size. Without the snap every
    // cell boundary jitters sub-cell as the camera moves, and cluster attributes
    // (centroid, cone, radiance) flicker frame to frame.
    const glm::vec3 eye = cam.position();
    active_ = std::clamp(cfg.cascades, 1, Config::kCascades);
    for (int l = 0; l < active_; ++l) {
        const float cell = cfg.cascade0_cell * std::pow(Config::kCascadeRatio, float(l));
        cascades_[l].cell = cell;
        const glm::vec3 snapped = glm::floor(eye / cell) * cell;
        cascades_[l].origin = snapped - glm::vec3(float(Config::kClipRes / 2) * cell);
    }

    const uint32_t total_cells = uint32_t(Config::kClipCells) * Config::kCascades;
    clear_uint_buffer(cell_count_);
    clear_uint_buffer(cell_mask_);

    // Pass 1: count + slot + occupancy bit, all cascades in one dispatch.
    live_pn_.bind_base(0);
    cell_count_.bind_base(1);
    surfel_slot_.bind_base(2);
    cell_mask_.bind_base(3);
    sub_[kCount]->begin();
    count_.use();
    set_cascade_uniforms(count_, cascades_);
    count_.set("u_live_count", live_count_);
    count_.set("u_cascades", active_);
    gl::dispatch_compute((live_count_ + 255) / 256, 1, 1);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    sub_[kCount]->end();

    // Pass 2: exclusive scan per cascade. Each cascade is exactly kClipCells =
    // 262144 elements, which is precisely the single-workgroup scan's limit.
    sub_[kScan]->begin();
    for (int l = 0; l < active_; ++l)
        scan_.run(cell_count_, cell_start_, uint32_t(Config::kClipCells), cascades_[l].cnt_off);
    // Per-cascade base offsets, so the compacted ranges of different cascades
    // do not overlap in packed_idx.
    cell_start_.bind_base(0);
    cell_count_.bind_base(1);
    cascade_base_.bind_base(2);
    bases_.use();
    bases_.set("u_cells", uint32_t(Config::kClipCells));
    bases_.set("u_cascades", active_);
    gl::dispatch_compute(1, 1, 1);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    sub_[kScan]->end();

    // Pass 3: scatter into cell order.
    live_pn_.bind_base(0);
    surfel_slot_.bind_base(1);
    cell_start_.bind_base(2);
    packed_idx_.bind_base(3);
    sub_[kCompact]->begin();
    cascade_base_.bind_base(5);
    compact_.use();
    set_cascade_uniforms(compact_, cascades_);
    compact_.set("u_live_count", live_count_);
    compact_.set("u_cascades", active_);
    gl::dispatch_compute((live_count_ + 255) / 256, 1, 1);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    sub_[kCompact]->end();

    // Pass 4: interleave start+count.
    cell_start_.bind_base(0);
    cell_count_.bind_base(1);
    cell_sc_.bind_base(2);
    cascade_base_.bind_base(3);
    sub_[kMerge]->begin();
    merge_.use();
    merge_.set("u_total_cells", uint32_t(Config::kClipCells) * uint32_t(active_));
    merge_.set("u_cells", uint32_t(Config::kClipCells));
    gl::dispatch_compute((total_cells + 255) / 256, 1, 1);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    sub_[kMerge]->end();

    // Pass 5: aggregate one cluster per non-empty cell.
    live_pn_.bind_base(0);
    live_alb_.bind_base(1);
    live_radiance_.bind_base(2);
    cell_sc_.bind_base(3);
    packed_idx_.bind_base(4);
    cluster_pn_.bind_base(5);
    cluster_rad_.bind_base(6);
    cell_mask_.bind_base(7);
    sub_[kCluster]->begin();
    clear_uint_buffer(stats_);
    clear_uint_buffer(far_cells_);
    clear_uint_buffer(cell_list_);

    // Compact the non-empty cells first, so the cluster pass is dispatched over
    // cells that have contents rather than over 262144 per cascade.
    cell_mask_.bind_base(0);
    cell_list_.bind_base(1);
    far_cells_.bind_base(2);
    celllist_.use();
    for (int l = 0; l < active_; ++l) {
        char name[64];
        std::snprintf(name, sizeof(name), "u_mask_offs[%d]", l);
        celllist_.set(name, cascades_[l].mask_off);
    }
    celllist_.set("u_cascades", active_);
    celllist_.set("u_coarsest", active_ - 1);
    celllist_.set("u_max_cells", kMaxListCells);
    celllist_.set("u_max_far", kMaxFarCells);
    gl::dispatch_compute((uint32_t(Config::kClipCells) + 255) / 256, 1, 1);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    cell_list_.bind_base(0);
    cluster_cmd_buf_.bind_base(1);
    cluster_cmd_.use();
    cluster_cmd_.set("u_max_cells", kMaxListCells);
    gl::dispatch_compute(1, 1, 1);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

    // One workgroup per non-empty cell.
    live_pn_.bind_base(0);
    live_radiance_.bind_base(2);
    cell_sc_.bind_base(3);
    packed_idx_.bind_base(4);
    cluster_pn_.bind_base(5);
    cluster_rad_.bind_base(6);
    stats_.bind_base(8);
    cell_list_.bind_base(9);
    cluster_.use();
    for (int l = 0; l < active_; ++l) {
        char name[64];
        std::snprintf(name, sizeof(name), "u_cnt_offs[%d]", l);
        cluster_.set(name, cascades_[l].cnt_off);
    }
    cluster_.set("u_max_cells", kMaxListCells);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, cluster_cmd_buf_.handle());
    gl::dispatch_compute_indirect(0);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    sub_[kCluster]->end();

    // Occupancy clipmap. Cleared and re-splatted per cascade, then mipped: the
    // box-filtered mip chain is the coverage field a cone march samples by cone
    // radius.
    sub_[kOccupancy]->begin();
    const float clear_zero = 0.0f;
    occupancy_build_.use();
    live_pn_.bind_base(0);
    for (int l = 0; l < active_; ++l) {
        glClearTexImage(occupancy_[l].handle(), 0, GL_RED, GL_FLOAT, &clear_zero);
        occupancy_[l].bind_image(0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8);
        // Same world volume as the cluster cascade, at kOccRes instead of
        // kClipRes.
        const float occ_cell = cascades_[l].cell / kOccScale;
        occupancy_build_.set("u_origin", cascades_[l].origin);
        occupancy_build_.set("u_inv_cell", 1.0f / occ_cell);
        occupancy_build_.set("u_res", Config::kOccRes);
        occupancy_build_.set("u_live_count", live_count_);
        gl::dispatch_compute((live_count_ + 255) / 256, 1, 1);
    }
    gl::memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // Mip chain by compute. glGenerateTextureMipmap on a 128^3 R8 volume costs
    // ~2.6 ms on this driver; the same downsample in compute is microseconds.
    occupancy_mip_.use();
    for (int l = 0; l < active_; ++l) {
        for (int m = 1; m < occ_mips_; ++m) {
            const int dst_res = std::max(1, Config::kOccRes >> m);
            occupancy_[l].bind_image(0, m - 1, GL_TRUE, 0, GL_READ_ONLY, GL_R8);
            occupancy_[l].bind_image(1, m, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8);
            occupancy_mip_.set("u_dst_res", dst_res);
            const GLuint g = GLuint((dst_res + 3) / 4);
            gl::dispatch_compute(g, g, g);
            gl::memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }
    }
    sub_[kOccupancy]->end();

    // Statistics are for the UI only, so read them back every ~30 frames rather
    // than stalling the pipeline every frame for three integers.
    if (--stats_countdown_ <= 0) {
        stats_countdown_ = 30;
        uint32_t counts[Config::kCascades] = {};
        glGetNamedBufferSubData(stats_.handle(), 0, sizeof(counts), counts);
        for (int l = 0; l < Config::kCascades; ++l) {
            occupied_[l] = (l < active_) ? counts[l] : 0;
            cluster_counts_[l] = counts[l];   // one cluster per non-empty cell
        }
    }
}

bool Clipmap::validate(const Config& cfg) {
    (void)cfg;
    if (live_count_ == 0) {
        gllib::log(gllib::LogLevel::warn, "clipmap validate: no live surfels");
        return false;
    }

    const uint32_t n = live_count_;
    const uint32_t cells = uint32_t(Config::kClipCells);
    const uint32_t total_cells = cells * Config::kCascades;
    const uint32_t mask_words = (cells + 31) / 32 * Config::kCascades;

    std::vector<glm::vec4> pn(size_t(n) * 2);
    std::vector<uint32_t> sc(size_t(total_cells) * 2);
    // packed_idx holds one entry per surfel PER CASCADE, not one per surfel.
    // Sizing this by live_count silently truncates every cascade after the
    // first and makes the comparison read past the end of the vector.
    const size_t packed_total = size_t(live_capacity_) * Config::kCascades;
    std::vector<uint32_t> idx(packed_total);
    std::vector<uint32_t> mask(mask_words);
    std::vector<glm::vec4> cpn(size_t(total_cells) * 2);
    std::vector<glm::vec4> crad(total_cells);

    glGetNamedBufferSubData(live_pn_.handle(), 0, GLsizeiptr(pn.size() * sizeof(glm::vec4)), pn.data());
    glGetNamedBufferSubData(cell_sc_.handle(), 0, GLsizeiptr(sc.size() * sizeof(uint32_t)), sc.data());
    glGetNamedBufferSubData(packed_idx_.handle(), 0, GLsizeiptr(idx.size() * sizeof(uint32_t)), idx.data());
    glGetNamedBufferSubData(cell_mask_.handle(), 0, GLsizeiptr(mask.size() * sizeof(uint32_t)), mask.data());
    glGetNamedBufferSubData(cluster_pn_.handle(), 0, GLsizeiptr(cpn.size() * sizeof(glm::vec4)), cpn.data());
    glGetNamedBufferSubData(cluster_rad_.handle(), 0, GLsizeiptr(crad.size() * sizeof(glm::vec4)), crad.data());

    // Sanity-check the live buffer itself before comparing anything derived from
    // it: a non-finite position silently fails every downstream comparison
    // (NaN < x and NaN > x are both false), so it would look like "not selected"
    // rather than like corrupt data.
    {
        size_t nonfinite = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const glm::vec4 p = pn[size_t(i) * 2];
            const glm::vec4 q = pn[size_t(i) * 2 + 1];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                !std::isfinite(p.w) || !std::isfinite(q.x) || !std::isfinite(q.y) ||
                !std::isfinite(q.z))
                ++nonfinite;
        }
        // Per-instance world-space bounds of the live surfels, so a mis-transformed
        // instance shows up as an obviously wrong box.
        for (size_t k = 0; k < cpu_instances_.size(); ++k) {
            const GpuInstance& g = cpu_instances_[k];
            glm::vec3 mn(1e30f), mx(-1e30f);
            float rmin = 1e30f, rmax = 0.0f;
            for (uint32_t j = 0; j < g.range.z && g.range.y + j < n; ++j) {
                const glm::vec4 p = pn[size_t(g.range.y + j) * 2];
                mn = glm::min(mn, glm::vec3(p));
                mx = glm::max(mx, glm::vec3(p));
                rmin = std::min(rmin, p.w);
                rmax = std::max(rmax, p.w);
            }
            gllib::logf(gllib::LogLevel::info,
                        "  live inst %2zu: pos [%.2f %.2f %.2f]..[%.2f %.2f %.2f] r %.4f..%.4f",
                        k, mn.x, mn.y, mn.z, mx.x, mx.y, mx.z, rmin, rmax);
        }
        if (nonfinite)
            gllib::logf(gllib::LogLevel::error, "  %zu live surfels have non-finite data",
                        nonfinite);
    }

    // CPU reference: a surfel belongs to EVERY cascade whose box contains it.
    std::vector<uint32_t> ref_count(total_cells, 0);
    std::vector<std::vector<uint32_t>> ref_members(total_cells);
    uint32_t outside = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const glm::vec3 p = glm::vec3(pn[size_t(i) * 2]);
        bool any = false;
        for (int l = 0; l < active_; ++l) {
            // Match the shader's arithmetic (multiply by the reciprocal) so a
            // surfel sitting exactly on a cell boundary does not land in
            // different cells on the two sides of the comparison.
            const float inv_cell = 1.0f / cascades_[l].cell;
            const glm::ivec3 c = glm::ivec3(glm::floor((p - cascades_[l].origin) * inv_cell));
            if (glm::any(glm::lessThan(c, glm::ivec3(0))) ||
                glm::any(glm::greaterThanEqual(c, glm::ivec3(Config::kClipRes)))) continue;
            any = true;
            const uint32_t ci = cascades_[l].cnt_off +
                                uint32_t(c.x + Config::kClipRes * (c.y + Config::kClipRes * c.z));
            ++ref_count[ci];
            ref_members[ci].push_back(i);
        }
        if (!any) ++outside;
    }

    size_t bad_count = 0, bad_members = 0, bad_mask = 0, bad_cluster = 0;
    for (uint32_t l = 0; l < uint32_t(active_); ++l) {
        for (uint32_t k = 0; k < cells; ++k) {
            const uint32_t ci = cascades_[l].cnt_off + k;
            const uint32_t gpu_count = sc[size_t(ci) * 2 + 1];
            const uint32_t gpu_start = sc[size_t(ci) * 2];
            if (gpu_count != ref_count[ci]) { ++bad_count; continue; }

            const bool gpu_bit = (mask[cascades_[l].mask_off + (k >> 5)] >> (k & 31)) & 1u;
            if (gpu_bit != (ref_count[ci] != 0)) ++bad_mask;
            if (ref_count[ci] == 0) continue;

            // Membership is order-independent: the compaction slot comes from an
            // atomic, so only the SET must match, not the sequence.
            if (size_t(gpu_start) + gpu_count > idx.size()) { ++bad_members; continue; }
            std::vector<uint32_t> got(idx.begin() + gpu_start,
                                      idx.begin() + gpu_start + gpu_count);
            std::vector<uint32_t> want = ref_members[ci];
            std::sort(got.begin(), got.end());
            std::sort(want.begin(), want.end());
            if (got != want) {
                if (bad_members < 2) {
                    std::string g, w;
                    for (size_t q = 0; q < std::min<size_t>(got.size(), 8); ++q)
                        g += std::to_string(got[q]) + " ";
                    for (size_t q = 0; q < std::min<size_t>(want.size(), 8); ++q)
                        w += std::to_string(want[q]) + " ";
                    gllib::logf(gllib::LogLevel::info,
                                "  cell L%u #%u start=%u count=%u\n    gpu:  %s\n    want: %s",
                                l, k, gpu_start, gpu_count, g.c_str(), w.c_str());
                }
                ++bad_members;
                continue;
            }

            // Cluster: area-weighted centroid and total projected area.
            glm::vec3 centre(0.0f);
            float area = 0.0f;
            for (uint32_t s : want) {
                const glm::vec4 pr = pn[size_t(s) * 2];
                const float a = 3.14159265f * pr.w * pr.w;
                centre += glm::vec3(pr) * a;
                area += a;
            }
            if (area <= 0.0f) continue;
            centre /= area;
            const glm::vec4 g_pn = cpn[size_t(ci) * 2];
            const float tol = std::max(cascades_[l].cell * 1e-3f, 1e-5f);
            if (glm::length(glm::vec3(g_pn) - centre) > tol ||
                std::abs(crad[ci].w - area) > area * 1e-3f + 1e-9f)
                ++bad_cluster;
        }
    }

    uint32_t bases[Config::kCascades] = {};
    glGetNamedBufferSubData(cascade_base_.handle(), 0, sizeof(bases), bases);
    uint32_t ref_total[Config::kCascades] = {};
    for (uint32_t l = 0; l < uint32_t(active_); ++l)
        for (uint32_t k = 0; k < cells; ++k)
            ref_total[l] += ref_count[cascades_[l].cnt_off + k];
    gllib::logf(gllib::LogLevel::info,
                "  cascade totals %u/%u/%u, gpu bases %u/%u/%u",
                ref_total[0], ref_total[1], ref_total[2], bases[0], bases[1], bases[2]);

    const bool ok = (bad_count | bad_members | bad_mask | bad_cluster) == 0;
    gllib::logf(ok ? gllib::LogLevel::info : gllib::LogLevel::error,
                "clipmap validate: %s  (%u live, %u outside all cascades; "
                "bad counts %zu, members %zu, mask %zu, clusters %zu)",
                ok ? "PASS" : "FAIL", n, outside,
                bad_count, bad_members, bad_mask, bad_cluster);
    return ok;
}

} // namespace mosaic

#pragma once

// ---------------------------------------------------------------------------
// S0 — instance assembly, and S1 — the camera-centred clipmaps.
//
// S0 transforms the object-space surfel sets of every visible instance into a
// flat world-space live buffer. Rigid motion is a matrix multiply here; the
// cache contents (SH irradiance, reservoirs) are indexed by LIVE id and carried
// across frames separately, which is why moving an object does not invalidate
// what it has learned.
//
// S1 builds three camera-centred 64^3 cascades of surfel CLUSTERS, rebuilt from
// scratch every frame. Rebuilding beats incremental update here: it makes
// dynamic geometry free, needs no sorting, and gives O(1) LOD selection by world
// distance — exactly the query the micro-render gather issues. It replaces the
// static octree of examples 36/37 and the radius-bucketed grids of 38.
//
// The grid build itself (count -> scan -> compact, with a free occupancy bitmask
// from the 0->1 transition of the per-cell atomic) follows example 38's kernels.
// ---------------------------------------------------------------------------

#include "gpu_util.hpp"
#include "mosaic.hpp"
#include "surfel_bake.hpp"

namespace mosaic {

// Per-instance record consumed by S0. Mirrors LiveInstance in
// shaders/common/clipmap.glsl.
struct GpuInstance {
    glm::mat4 xform;        // object -> world
    glm::vec4 aabb_min;     // xyz = set AABB min,    w = radius_scale
    glm::vec4 aabb_ext;     // xyz = set AABB extent, w = object_scale
    glm::uvec4 range;       // src_base, dst_base, count, flags
};
static_assert(sizeof(GpuInstance) == 64 + 16 + 16 + 16, "GpuInstance layout");

// One cascade of the cluster clipmap.
struct Cascade {
    glm::vec3 origin{0.0f};   // world position of cell (0,0,0)'s min corner
    float cell = 1.0f;
    uint32_t cnt_off = 0;     // segment offsets into the combined buffers
    uint32_t mask_off = 0;
    uint32_t live_count = 0;  // surfels assigned to this cascade
};

class Clipmap {
public:
    bool init();
    bool poll();

    // S0: assemble the live world-space surfel set for this frame.
    // Returns the number of live surfels.
    uint32_t assemble(SurfelLibrary& lib,
                      const std::vector<Instance>& instances,
                      const gfx::Camera& cam,
                      const Config& cfg,
                      bool freeze_lod);

    // S1: rebuild the three cascades around the camera.
    void build(const gfx::Camera& cam, const Config& cfg);

    uint32_t live_count() const { return live_count_; }
    int active_cascades() const { return active_; }
    const Cascade& cascade(int i) const { return cascades_[i]; }
    uint32_t cluster_count(int i) const { return cluster_counts_[i]; }
    uint32_t occupied_cells(int i) const { return occupied_[i]; }
    // Instances that did not fit the live budget even at the coarsest LOD.
    uint32_t budget_dropped() const { return budget_dropped_; }

    // Live surfel SoA, bound by every later stage.
    gl::Buffer& live_pn() { return live_pn_; }          // 2 x vec4 per surfel
    gl::Buffer& live_alb() { return live_alb_; }        // vec4(albedo, emissive_lum)
    gl::Buffer& live_emissive() { return live_emissive_; }  // vec4(emissive, 0)
    gl::Buffer& live_meta() { return live_meta_; }      // uint: flags | instance << 8
    gl::Buffer& live_src() { return live_src_; }        // uint: source surfel index
    gl::Buffer& live_radiance() { return live_radiance_; }  // vec4: outgoing radiance (S2)

    // Cascade cell tables: uvec2(start, count) per cell, and the compacted
    // per-cell surfel id lists.
    gl::Buffer& cell_sc() { return cell_sc_; }
    gl::Buffer& cell_mask() { return cell_mask_; }
    gl::Buffer& packed_idx() { return packed_idx_; }

    // Aggregated clusters, one per non-empty cell.
    gl::Buffer& cluster_pn() { return cluster_pn_; }    // 2 x vec4: (centroid, radius), (cone axis, cos half-angle)
    gl::Buffer& cluster_rad() { return cluster_rad_; }  // vec4(outgoing radiance, projected area)

    // Occupancy clipmap: one R8 3D texture per cascade, mipped so a cone march
    // can select a level by cone radius.
    const gl::Texture& occupancy(int i) const { return occupancy_[i]; }
    int occupancy_mips() const { return occ_mips_; }

    // Compacted non-empty cells of the coarsest cascade: [0] = count,
    // [1..] = linear cell indices. The gather visits all of them.
    gl::Buffer& far_cells() { return far_cells_; }
    static constexpr uint32_t kMaxFarCells = 16384;
    // Upper bound on non-empty cells across all active cascades.
    static constexpr uint32_t kMaxListCells = 512 * 1024;

    // Per-instance LOD chosen this frame, for the UI.
    const std::vector<int>& instance_lod() const { return instance_lod_; }

    // Sub-pass breakdown of build(), so a regression inside S1 is attributable
    // without re-instrumenting.
    enum SubPass { kCount, kScan, kCompact, kMerge, kCluster, kOccupancy, kSubPassCount };
    static const char* subpass_name(int i);
    PassTimer& subpass(int i) { return *sub_[i]; }

    // Sum of the sub-pass GPU times. Only one GL_TIME_ELAPSED query may be
    // active at a time, so S1 cannot be wrapped in an outer GPU query while its
    // sub-passes are timed; the sum is the outer figure.
    double gpu_ms() const {
        double t = 0.0;
        for (int i = 0; i < kSubPassCount; ++i) t += sub_[i]->disp_gpu();
        return t;
    }

    // Rebuilds this frame's cascade assignment on the CPU from the live surfel
    // buffer and compares it against the GPU result: per-cell counts, the
    // compacted membership lists, the occupancy bitmask, and the aggregated
    // cluster centroid/area. Returns true when everything matches. One-shot and
    // stalling; drive it from MOSAIC_VALIDATE, never per frame.
    bool validate(const Config& cfg);

private:
    void resize_buffers(uint32_t live_capacity);

    Pipeline assemble_;
    Pipeline count_;
    Pipeline bases_;
    Pipeline compact_;
    Pipeline merge_;
    Pipeline celllist_;
    Pipeline cluster_cmd_;
    Pipeline cluster_;
    Pipeline occupancy_build_;
    Pipeline occupancy_mip_;
    PrefixSum scan_;

    gl::Buffer instances_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer live_pn_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer live_alb_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer live_emissive_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer live_meta_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer live_src_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer live_radiance_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};

    gl::Buffer cell_count_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer cell_start_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer cell_sc_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer cell_mask_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer surfel_slot_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer packed_idx_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer cascade_base_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};

    gl::Buffer cluster_pn_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer cluster_rad_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer far_cells_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer cell_list_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer cluster_cmd_buf_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};

    gl::Texture occupancy_[Config::kCascades]{
        gl::Texture(gl::TextureType::tex_3d),
        gl::Texture(gl::TextureType::tex_3d),
        gl::Texture(gl::TextureType::tex_3d),
    };
    int occ_mips_ = 1;

    // Per-cascade non-empty cell counts, read back occasionally for the UI.
    gl::Buffer stats_{gl::BufferType::shader, gl::BufferUsage::dynamic_read};
    int stats_countdown_ = 0;

    std::unique_ptr<PassTimer> sub_[kSubPassCount];

    Cascade cascades_[Config::kCascades];
    uint32_t cluster_counts_[Config::kCascades] = {};
    uint32_t occupied_[Config::kCascades] = {};
    uint32_t live_count_ = 0;
    int active_ = Config::kCascades;
    bool log_instances_ = false;

public:
    void log_instances_once() { log_instances_ = true; }
private:
    uint32_t budget_dropped_ = 0;
    uint32_t live_capacity_ = 0;
    std::vector<GpuInstance> cpu_instances_;
    std::vector<int> instance_lod_;
};

} // namespace mosaic
